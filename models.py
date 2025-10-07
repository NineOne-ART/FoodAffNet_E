import torch
import torch.nn as nn
import torch.nn.functional as F
import os


# ---------------------------
# 多头自注意力模块
# ---------------------------
class SelfAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(SelfAttentionBlock, self).__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_len, embed_dim]
        attn_output, _ = self.attn(x, x, x)
        x = self.norm(x + self.dropout(attn_output))
        return x


# ---------------------------
# 三层 1D 卷积模块
# ---------------------------
class Conv1DBlock(nn.Module):
    def __init__(self, in_channels, out_channels=128, dropout=0.3):
        super(Conv1DBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(in_channels, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(128, out_channels, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.AdaptiveMaxPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch, in_channels, seq_len]
        x = self.conv(x)
        x = self.pool(x).squeeze(-1)  # [batch, out_channels]
        x = self.dropout(x)
        return x


# ---------------------------
# Transformer 编码模块
# ---------------------------
class TransformerEncoderBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4, hidden_dim=256, dropout=0.1, num_layers=2):
        super(TransformerEncoderBlock, self).__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, hidden_dim, dropout, batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x shape: [batch, seq_len, embed_dim]
        x = self.encoder(x)
        x = x.transpose(1, 2)  # for pooling
        x = self.pool(x).squeeze(-1)
        x = self.dropout(x)
        return x


# ---------------------------
# 小分子特征提取器（无图网络）
# ---------------------------
class MoleculeFeatureExtractor(nn.Module):
    def __init__(self, input_dim=78, seq_length=100, dropout=0.3):
        super(MoleculeFeatureExtractor, self).__init__()
        self.seq_length = seq_length
        self.input_dim = input_dim

        # 通道1: 自注意力
        self.attn_block = SelfAttentionBlock(embed_dim=input_dim)

        # 通道2: 卷积
        self.conv_branch = Conv1DBlock(in_channels=input_dim, out_channels=128)

        # 特征投影
        self.projection = nn.Linear(input_dim + 128, 256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, seq_length, input_dim]
        batch_size = x.size(0)

        # 自注意力通道
        attn_out = self.attn_block(x)  # [batch, seq_length, input_dim]
        attn_pooled = attn_out.mean(dim=1)  # [batch, input_dim]

        # 卷积通道
        conv_input = x.transpose(1, 2)  # [batch, input_dim, seq_length]
        conv_out = self.conv_branch(conv_input)  # [batch, 128]

        # 融合特征
        combined = torch.cat([attn_pooled, conv_out], dim=1)  # [batch, input_dim + 128]
        projected = F.relu(self.projection(combined))
        projected = self.dropout(projected)

        return projected


class ProteinFeatureExtractor(nn.Module):
    def __init__(self, embed_dim=320, dropout=0.3):
        super(ProteinFeatureExtractor, self).__init__()

        # 通道1: Transformer
        self.transformer = TransformerEncoderBlock(embed_dim=embed_dim)

        # 通道2: 卷积
        self.conv_branch = Conv1DBlock(in_channels=embed_dim, out_channels=128)

        # 特征投影
        self.projection = nn.Linear(embed_dim + 128, 256)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, embed_dim] - ESM均值池化后的特征

        # Transformer通道
        seq_representation = x.unsqueeze(1).repeat(1, 16, 1)  # [batch, 16, embed_dim]
        trans_out = self.transformer(seq_representation)  # [batch, embed_dim]

        # 卷积通道
        conv_input = x.unsqueeze(-1).transpose(1, 2)  # [batch, embed_dim, 1]
        conv_out = self.conv_branch(conv_input)  # [batch, 128]

        # 融合特征
        combined = torch.cat([trans_out, conv_out], dim=1)  # [batch, embed_dim + 128]
        projected = F.relu(self.projection(combined))
        projected = self.dropout(projected)

        return projected


class AttentionConvModel(nn.Module):
    def __init__(self, n_output=1, mol_input_dim=78, dropout=0.3):
        super(AttentionConvModel, self).__init__()

        self.dropout = dropout
        self.n_output = n_output

        # 小分子特征提取
        self.mol_extractor = MoleculeFeatureExtractor(
            input_dim=mol_input_dim,
            dropout=dropout
        )

        # 蛋白质特征提取
        self.prot_extractor = ProteinFeatureExtractor(
            embed_dim=320,
            dropout=dropout
        )

        # 融合层 - 输出均值
        self.fusion_fc1 = nn.Linear(512, 512)
        self.fusion_fc2 = nn.Linear(512, 256)
        self.out_mean = nn.Linear(256, n_output)

        # 方差预测头
        self.var_fc1 = nn.Linear(512, 256)
        self.var_fc2 = nn.Linear(256, 128)
        self.out_var = nn.Linear(128, n_output)

        # 确保方差为正
        self.softplus = nn.Softplus()

    def forward(self, data):
        # 小分子特征提取
        mol_features = data['compound_sequence']
        mol_embedding = self.mol_extractor(mol_features)

        # 蛋白质特征提取
        prot_embed = data['protein_embedding']
        prot_embedding = self.prot_extractor(prot_embed)

        # 特征融合
        fusion = torch.cat([mol_embedding, prot_embedding], dim=1)

        # 均值预测
        mean_fusion = F.relu(self.fusion_fc1(fusion))
        mean_fusion = F.dropout(mean_fusion, p=self.dropout, training=self.training)
        mean_fusion = F.relu(self.fusion_fc2(mean_fusion))
        mean_fusion = F.dropout(mean_fusion, p=self.dropout, training=self.training)
        mean = self.out_mean(mean_fusion)

        # 方差预测
        var_fusion = F.relu(self.var_fc1(fusion))
        var_fusion = F.dropout(var_fusion, p=self.dropout, training=self.training)
        var_fusion = F.relu(self.var_fc2(var_fusion))
        var_fusion = F.dropout(var_fusion, p=self.dropout, training=self.training)
        var = self.softplus(self.out_var(var_fusion)) + 1e-6  # 确保方差为正且不为零

        return mean, var


class SimpleAttentionConvModel(nn.Module):
    def __init__(self, n_output=1, mol_dim=78, prot_dim=320, dropout=0.3):
        super(SimpleAttentionConvModel, self).__init__()

        # 小分子处理
        self.mol_attention = SelfAttentionBlock(mol_dim)
        self.mol_conv = Conv1DBlock(mol_dim)
        self.mol_fc = nn.Linear(mol_dim + 128, 128)

        # 蛋白质处理
        self.prot_attention = SelfAttentionBlock(prot_dim)
        self.prot_conv = Conv1DBlock(prot_dim)
        self.prot_fc = nn.Linear(prot_dim + 128, 128)

        # 融合预测 - 均值
        self.fc1 = nn.Linear(256, 128)
        self.fc2 = nn.Linear(128, 64)
        self.out_mean = nn.Linear(64, n_output)

        # 方差预测
        self.var_fc1 = nn.Linear(256, 64)
        self.var_fc2 = nn.Linear(64, 32)
        self.out_var = nn.Linear(32, n_output)

        self.dropout = nn.Dropout(dropout)
        self.softplus = nn.Softplus()

    def forward(self, data):
        # 提取数据
        mol_seq = data['compound_sequence']
        prot_embed = data['protein_embedding']

        # 小分子处理
        mol_attn = self.mol_attention(mol_seq).mean(dim=1)
        mol_conv = self.mol_conv(mol_seq.transpose(1, 2))
        mol_out = F.relu(self.mol_fc(torch.cat([mol_attn, mol_conv], dim=1)))

        # 蛋白质处理
        prot_seq = prot_embed.unsqueeze(1).repeat(1, 16, 1)
        prot_attn = self.prot_attention(prot_seq).mean(dim=1)
        prot_conv = self.prot_conv(prot_embed.unsqueeze(-1).transpose(1, 2))
        prot_out = F.relu(self.prot_fc(torch.cat([prot_attn, prot_conv], dim=1)))

        # 融合
        combined = torch.cat([mol_out, prot_out], dim=1)

        # 均值预测
        mean_x = F.relu(self.fc1(combined))
        mean_x = self.dropout(mean_x)
        mean_x = F.relu(self.fc2(mean_x))
        mean_x = self.dropout(mean_x)
        mean = self.out_mean(mean_x)

        # 方差预测
        var_x = F.relu(self.var_fc1(combined))
        var_x = self.dropout(var_x)
        var_x = F.relu(self.var_fc2(var_x))
        var_x = self.dropout(var_x)
        var = self.softplus(self.out_var(var_x)) + 1e-6

        return mean, var
