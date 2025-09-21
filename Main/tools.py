from __future__ import annotations
import os
import sys
import numpy as np
import torch
from scipy.special import gamma
from contextlib import contextmanager

@contextmanager
def suppress_stdout():
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

def log(s, path, islog):
    if islog:
        with open(path, 'a') as f:
            f.write(s+'\n')
    else:
        print(s)

def generate_Empty(repeats, testN):
    CS, COV, SIZE = np.zeros((repeats, testN, 2)), np.zeros((repeats, testN)), np.zeros((repeats, testN))
    return CS, COV, SIZE

def generate_Empty_Cl(repeats, testN, typeNum):
    CS, COV, SIZE = np.zeros((repeats, testN, typeNum)), np.zeros((repeats, testN)), np.zeros((repeats, testN))
    return CS, COV, SIZE

def setseed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)

def checkDir(path):
    if not os.path.exists(path):
        try:
            os.mkdir(path)
        except:
            pass

def relog(path):
    with open(path, 'w') as f:
        f.write('')

def hypersphere_volume(d, r):
    """
    计算d维超球半径为r的体积
    :param d: 维度 (标量或数组)
    :param r: 半径 (标量或数组)
    :return: 体积 (标量或数组)
    """
    numerator = (np.pi ** (d / 2)) * (r ** d)
    denominator = gamma(d / 2 + 1)
    return numerator / denominator

def multivariate_normal_density(x, mu, sigma):
    """
    计算d维正态分布的密度函数
    :param x: 数据点 (n x d)
    :param mu: 均值向量 (1 x d)
    :param sigma: 标准差 (标量)
    :return: 密度值 (n,)
    """
    n, d = x.shape
    # 计算常数项
    const = 1 / ((2 * np.pi * sigma**2) ** (d / 2))
    # 计算指数项
    diff = x - mu.reshape((1,-1))  # n x d
    squared_norm = np.sum(diff**2, axis=1)  # n,
    exp_term = np.exp(-squared_norm / (2 * sigma**2))
    # 返回密度
    return const * exp_term

def wasserstein_distance(a:np.ndarray, b:np.ndarray):
    sorted_a = np.sort(a)
    sorted_b = np.sort(b)
    return np.mean(np.abs(sorted_a - sorted_b))

def selectFromIndice(X, Y, requiredIndice, restIndice):
    requiredX, requiredY = X[requiredIndice], Y[requiredIndice]
    restX, restY = X[restIndice], Y[restIndice]
    return requiredX, requiredY, restX, restY

def one_summation(COV, SIZE, alpha:float=.1):
    mar = np.mean(COV)
    size = np.mean(SIZE)
    local_cov = np.mean(np.abs(COV-(1-alpha)))
    return mar, size, local_cov

def summation(COV, SIZE, alpha:float=.1):
    """
    :param COV: repeats, testN
    :param SIZE: repeats, testN
    """
    mar = np.mean(COV)
    size = np.mean(SIZE)
    local_mar = COV.mean(0)
    local_cov = np.mean(np.abs(local_mar-(1-alpha)))
    tt_local_cov = np.mean(np.abs(COV-(1-alpha)))
    return mar, size, local_cov, tt_local_cov

def empiricalQuantile(X:np.ndarray, W:np.ndarray, q:float=.9):
    indice = np.argsort(X, axis=1)
    rowindice = np.repeat(np.array(range(indice.shape[0])), W.shape[1], axis=0)
    rowsorted_X = X[rowindice, indice.reshape(-1)].reshape((X.shape[0], X.shape[1]))
    rowsorted_W = W[rowindice, indice.reshape(-1)].reshape((W.shape[0], W.shape[1]))
    cumsum_W = np.cumsum(rowsorted_W, axis=1)
    mask = cumsum_W > q
    re_X = rowsorted_X * mask + (1-mask) * (np.max(X)+1)
    quantile = np.min(re_X, axis=1)
    return quantile

def saveData(path, CS, COV, SIZE, name):
    np.save(f"{path}/{name}_CS.npy", CS)
    np.save(f"{path}/{name}_COV.npy", COV)
    np.save(f"{path}/{name}_SIZE.npy", SIZE)

def loadData(path, name):
    CS = np.load(f"{path}/{name}_CS.npy")
    COV = np.load(f"{path}/{name}_COV.npy")
    SIZE = np.load(f"{path}/{name}_SIZE.npy")
    return CS, COV, SIZE

class unitWeight:
    def __init__(self):
        pass
    def ratio(self, X:np.ndarray):
        return np.ones(X.shape[0], dtype=float).reshape((-1,1))
    def ratioT(self, X):
        return torch.ones(X.shape[0]).float().view(-1,1)

class wrapFederatedDensityRatio:
    def __init__(self, dre):
        self.dre = dre
    def ratio(self, X):
        return self.dre.cali_predict(X)
    def ratioT(self, X):
        return torch.tensor(self.dre.cali_predict(X.numpy())).float()

class CDF:
    def __init__(self):
        pass

    def conditional_cdf(self, x:np.ndarray, y:np.ndarray):
        if len(y.shape) == 1:
            y = y.reshape(-1, 1)
        return np.ones((x.shape[0], y.shape[1]))