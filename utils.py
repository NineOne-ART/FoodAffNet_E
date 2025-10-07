import numpy as np
from math import sqrt
from scipy import stats

def rmse(y, f):
    """计算均方根误差"""
    rmse = sqrt(((y - f) ** 2).mean(axis=0))
    return rmse

def mse(y, f):
    """计算均方误差"""
    mse = ((y - f) ** 2).mean(axis=0)
    return mse

def pearson(y, f):
    """计算皮尔逊相关系数"""
    rp = np.corrcoef(y, f)[0, 1]
    return rp

def spearman(y, f):
    """计算斯皮尔曼相关系数"""
    rs = stats.spearmanr(y, f)[0]
    return rs

def ci(y, f):
    """计算一致性指数 (Concordance Index)"""
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y) - 1
    j = i - 1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z + 1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i - 1
    ci = S / z if z > 0 else 0.5
    return ci