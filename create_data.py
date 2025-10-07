import os
import json
import pickle
import pandas as pd
import numpy as np
from rdkit import Chem
import torch
from esm import pretrained
from collections import OrderedDict
from utils import *


# -----------------------------
# 小分子序列化特征
# -----------------------------
def smile_to_sequence_features(smile, max_length=100):
    """
    将SMILES转换为序列特征
    返回: 固定长度的特征序列 [max_length, 78]
    """
    mol = Chem.MolFromSmiles(smile)
    if mol is None:
        return np.zeros((max_length, 78))

    features = []
    for atom in mol.GetAtoms():
        feature = atom_features(atom)
        features.append(feature / sum(feature))

    # 填充或截断到固定长度
    if len(features) > max_length:
        features = features[:max_length]
    else:
        # 填充零向量
        padding = [np.zeros(78) for _ in range(max_length - len(features))]
        features.extend(padding)

    return np.array(features)


def atom_features(atom):
    return np.array(one_of_k_encoding_unk(atom.GetSymbol(),
                                          ['C', 'N', 'O', 'S', 'F', 'Si', 'P', 'Cl', 'Br', 'Mg', 'Na', 'Ca', 'Fe', 'As',
                                           'Al', 'I', 'B', 'V', 'K', 'Tl', 'Yb', 'Sb', 'Sn', 'Ag', 'Pd', 'Co', 'Se',
                                           'Ti', 'Zn', 'H', 'Li', 'Ge', 'Cu', 'Au', 'Ni', 'Cd', 'In', 'Mn', 'Zr', 'Cr',
                                           'Pt', 'Hg', 'Pb', 'Unknown']) +
                    one_of_k_encoding(atom.GetDegree(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetTotalNumHs(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    one_of_k_encoding_unk(atom.GetImplicitValence(), [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]) +
                    [atom.GetIsAromatic()])


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(x, allowable_set))
    return list(map(lambda s: x == s, allowable_set))


def one_of_k_encoding_unk(x, allowable_set):
    if x not in allowable_set:
        x = allowable_set[-1]
    return list(map(lambda s: x == s, allowable_set))


# -----------------------------
# ESM 蛋白质嵌入生成
# -----------------------------
print("Loading ESM model for protein embedding...")

local_esm_path = "C:/Users/S/Desktop/FoodAffNet/4.2/esm2_t6_8M_UR50D.pt"

try:
    if os.path.exists(local_esm_path):
        print(f"Found local ESM model at {local_esm_path}, loading...")
        esm_model, alphabet = pretrained.load_model_and_alphabet_local(local_esm_path)
    else:
        print("Local ESM model not found, loading from pretrained weights (online)...")
        esm_model, alphabet = pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")
except Exception as e:
    print(f"⚠️ Failed to load local ESM model: {e}")
    print("Falling back to online pretrained model...")
    esm_model, alphabet = pretrained.load_model_and_alphabet("esm2_t6_8M_UR50D")

esm_model.eval()
batch_converter = alphabet.get_batch_converter()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
esm_model = esm_model.to(device)


def esm_embed(protein_sequences, esm_model, batch_converter, device='cpu'):
    """使用 ESM 模型生成蛋白质嵌入（均值池化）"""
    embeddings = []
    with torch.no_grad():
        for i in range(0, len(protein_sequences), 8):
            batch_data = [("protein", seq) for seq in protein_sequences[i:i + 8]]
            _, _, batch_tokens = batch_converter(batch_data)
            batch_tokens = batch_tokens.to(device)
            results = esm_model(batch_tokens, repr_layers=[6])
            layer_6_output = results["representations"][6]
            mean_embeddings = layer_6_output.mean(dim=1).cpu().numpy()
            embeddings.extend(mean_embeddings)
    return np.array(embeddings)


# -----------------------------
# 数据处理主流程
# -----------------------------
all_prots = []
datasets = ['kiba', 'davis']
for dataset in datasets:
    print('Convert data from DeepDTA for ', dataset)
    fpath = 'data/' + dataset + '/'
    train_fold = json.load(open(fpath + "folds/train_fold_setting1.txt"))
    train_fold = [ee for e in train_fold for ee in e]
    valid_fold = json.load(open(fpath + "folds/test_fold_setting1.txt"))
    ligands = json.load(open(fpath + "ligands_can.txt"), object_pairs_hook=OrderedDict)
    proteins = json.load(open(fpath + "proteins.txt"), object_pairs_hook=OrderedDict)
    affinity = pickle.load(open(fpath + "Y", "rb"), encoding='latin1')
    drugs = []
    prots = []
    for d in ligands.keys():
        lg = Chem.MolToSmiles(Chem.MolFromSmiles(ligands[d]), isomericSmiles=True)
        drugs.append(lg)
    for t in proteins.keys():
        prots.append(proteins[t])
    if dataset == 'davis':
        affinity = [-np.log10(y / 1e9) for y in affinity]
    affinity = np.asarray(affinity)
    opts = ['train', 'test']
    for opt in opts:
        rows, cols = np.where(np.isnan(affinity) == False)
        if opt == 'train':
            rows, cols = rows[train_fold], cols[train_fold]
        elif opt == 'test':
            rows, cols = rows[valid_fold], cols[valid_fold]
        with open('data/' + dataset + '_' + opt + '.csv', 'w') as f:
            f.write('compound_iso_smiles,target_sequence,affinity\n')
            for pair_ind in range(len(rows)):
                ls = []
                ls += [drugs[rows[pair_ind]]]
                ls += [prots[cols[pair_ind]]]
                ls += [affinity[rows[pair_ind], cols[pair_ind]]]
                f.write(','.join(map(str, ls)) + '\n')
    print('\ndataset:', dataset)
    print('train_fold:', len(train_fold))
    print('test_fold:', len(valid_fold))
    print('len(set(drugs)), len(set(prots)):', len(set(drugs)), len(set(prots)))
    all_prots += list(set(prots))

compound_iso_smiles = []
for dt_name in ['kiba', 'davis']:
    opts = ['train', 'test']
    for opt in opts:
        df = pd.read_csv('data/' + dt_name + '_' + opt + '.csv')
        compound_iso_smiles += list(df['compound_iso_smiles'])
compound_iso_smiles = set(compound_iso_smiles)

# 创建序列特征字典
smile_sequence_features = {}
for smile in compound_iso_smiles:
    seq_features = smile_to_sequence_features(smile)
    smile_sequence_features[smile] = seq_features

# -----------------------------
# 创建新的数据集类
# -----------------------------
from torch.utils.data import Dataset
import torch


class SequenceDataset(Dataset):
    def __init__(self, root, dataset, xd, xt, y, smile_sequence_features):
        super(SequenceDataset, self).__init__()
        self.dataset = dataset
        self.smile_sequence_features = smile_sequence_features

        # 处理小分子特征
        self.compounds = xd
        self.compound_features = []
        for smile in xd:
            features = smile_sequence_features[smile]
            self.compound_features.append(features)

        # 处理蛋白质特征（将在后续生成ESM嵌入）
        self.proteins = xt
        self.affinities = y

        # 保存为pt文件
        self.processed_dir = os.path.join(root, 'processed')
        os.makedirs(self.processed_dir, exist_ok=True)
        self.processed_file = os.path.join(self.processed_dir, f'{dataset}.pt')

        # 如果文件不存在，则处理数据
        if not os.path.isfile(self.processed_file):
            self.process()
        else:
            self.data = torch.load(self.processed_file)

    def process(self):
        print(f'Processing {self.dataset} dataset...')

        # 生成蛋白质ESM嵌入
        print(f"Generating ESM embeddings for {self.dataset}...")
        prot_embeddings = esm_embed(self.proteins, esm_model, batch_converter, device)

        data_list = []
        for i in range(len(self.compounds)):
            compound_seq = torch.FloatTensor(self.compound_features[i])
            protein_embed = torch.FloatTensor(prot_embeddings[i])
            affinity = torch.FloatTensor([self.affinities[i]])

            data = {
                'compound_sequence': compound_seq,
                'protein_embedding': protein_embed,
                'affinity': affinity
            }
            data_list.append(data)

        self.data = data_list
        torch.save(data_list, self.processed_file)
        print(f'Saved processed data to {self.processed_file}')

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


# -----------------------------
# 转换为 PyTorch 数据格式
# -----------------------------
datasets = ['davis', 'kiba']
for dataset in datasets:
    processed_data_file_train = f'data/processed/{dataset}_train_sequence.pt'
    processed_data_file_test = f'data/processed/{dataset}_test_sequence.pt'

    if ((not os.path.isfile(processed_data_file_train)) or (not os.path.isfile(processed_data_file_test))):
        # 训练数据
        df_train = pd.read_csv(f'data/{dataset}_train.csv')
        train_drugs = list(df_train['compound_iso_smiles'])
        train_prots = list(df_train['target_sequence'])
        train_Y = list(df_train['affinity'])

        # 测试数据
        df_test = pd.read_csv(f'data/{dataset}_test.csv')
        test_drugs = list(df_test['compound_iso_smiles'])
        test_prots = list(df_test['target_sequence'])
        test_Y = list(df_test['affinity'])

        print(f'Creating sequence dataset for {dataset} train...')
        train_data = SequenceDataset(
            root='data',
            dataset=f'{dataset}_train_sequence',
            xd=train_drugs,
            xt=train_prots,
            y=train_Y,
            smile_sequence_features=smile_sequence_features
        )

        print(f'Creating sequence dataset for {dataset} test...')
        test_data = SequenceDataset(
            root='data',
            dataset=f'{dataset}_test_sequence',
            xd=test_drugs,
            xt=test_prots,
            y=test_Y,
            smile_sequence_features=smile_sequence_features
        )

        print(f'{processed_data_file_train} and {processed_data_file_test} have been created')
    else:
        print(f'{processed_data_file_train} and {processed_data_file_test} are already created')

print("数据处理完成：小分子使用序列特征，蛋白质使用ESM嵌入")

