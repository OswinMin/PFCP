import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Union
import scipy.stats as stats
from scipy.stats import rankdata
from Agent import *
from tools import *
from Predictor import *
from FDRE import *
from FEng import *
from FCP import *
from FedCP import *
from FedCPQQ import *
from FClus import *
import warnings
import datetime
warnings.filterwarnings('ignore')

def sigma(X, gamma):
    if len(X.shape) == 1:
        X = X.reshape((-1, 1))
    return (np.abs((np.abs(X)-(2/np.pi)**0.5).sum(-1)) / X.shape[1]**0.5).reshape((-1, 1)) * np.sqrt(gamma)

def generateAgent(gamma, n, d, me):
    X = np.random.normal(0, 1, (n, d))
    Y = X.sum(-1).reshape((-1, 1)) / me + np.random.normal(0, 1, (n, 1)) * sigma(X, gamma)
    agent = Agent(X, Y)
    class CDF_(CDF):
        def __init__(self):
            super().__init__()
        def conditional_cdf(self, x:np.ndarray, y:np.ndarray):
            if len(y.shape) == 1:
                y = y.reshape(-1, 1)
            y = (y - x.sum(-1).reshape((-1, 1)) / me) / sigma(X, gamma)    # regularized y
            return stats.norm.cdf(y)
    cdf = CDF_()
    agent.loadCDF(cdf)
    return agent

def logResult(mar, size, local_cov, tt_local_cov, name, isLog=False, path='', log=log):
    log(f"{name.upper()} : marginal : {mar:.4f}, mean size : {size:.4f}, conditional cov error : {local_cov:.4f}, CTA error : {tt_local_cov:.4f}", path=path, islog=isLog)

def logResult_(mar, size, local_cov, name, isLog=False, path='', log=log):
    log(f"{name.upper()} : marginal : {mar:.4f}, mean size : {size:.4f}, CTA error : {local_cov:.4f}", path=path, islog=isLog)

def logInfo(dic, Lpath):
    for k, v in dic.items():
        log(f"{k} : {v}", path=Lpath, islog=True)

def sampleGamma(agentN, cor_ratio:float=0.5):
    cor_N = np.random.binomial(agentN, cor_ratio)
    gamma = np.zeros(agentN)
    gamma[:cor_N] = np.random.uniform(0, 0.2, cor_N)
    gamma[cor_N:] = np.random.uniform(0.6,1., agentN-cor_N)
    return gamma

def predict_sum(fcp, test_agent, name, isLog, Lpath, log):
    cs = fcp.predict(test_agent.getX())
    cov = test_agent.coverage(test_agent.getX(), cs).reshape(-1)
    size = cs[:, 1] - cs[:, 0]
    mar0, size0, local_cov0 = one_summation(cov, size, .1)
    logResult_(mar0, size0, local_cov0, name, isLog=isLog, path=Lpath, log=log)
    return cs, cov, size

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != 'Sele_Aux':
        os.chdir(os.path.join(os.getcwd(), 'Sele_Aux'))
    # each agent has 3*n samples
    d, n, agentN, cor_ratio, seed = 20, 100, 40, 0.5, 1
    if len(sys.argv) > 1:
        d, n, agentN, cor_ratio, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), int(sys.argv[5])
        isLog = True
    else:
        isLog = False

    Rpath = f"../Result/Sam_abs/{d}_{n}_{agentN}_{cor_ratio}"
    Lpath = f"../Log/Sam_abs/{d}_{n}_{agentN}_{cor_ratio}.txt"
    checkDir("../Result")
    checkDir("../Result/Sam_abs")
    checkDir(Rpath)
    checkDir("../Log")
    checkDir("../Log/Sam_abs")

    epoches = 300
    repeats = 100
    testN = 1000
    hidden_dims_pred = [30, 30]
    hidden_dims_dre = [30]
    num_layer, hidden_dim, noise_dim = 2, 100, d
    if isLog:
        relog(Lpath)
        dic = {'d':d, 'n':n, 'agentN':agentN, 'cor_ratio':cor_ratio, 'seed':seed, 'epoches':epoches, 'repeats':repeats, 'testN':testN, 'hidden_dims_pred':hidden_dims_pred, 'hidden_dims_dre':hidden_dims_dre, "num_layer":num_layer, "hidden_dim":hidden_dim, "noise_dim":noise_dim}
        logInfo(dic, Lpath)

    setseed(seed)
    test_agent = generateAgent(1, testN, d, 1)
    res_matrix = np.zeros((repeats, agentN))

    for rep in range(repeats):
        seed_rep = (rep + 1) * (np.abs(seed) + 1)
        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"#"*20+f" Repeat {rep+1} "+f"#"*20, path=Lpath, islog=isLog)
        log(f"Start at time {tn}", path=Lpath, islog=isLog)

        setseed(seed_rep)
        predTrAgent0 = generateAgent(1, n, d, 1)
        engTrAgent0 = generateAgent(1, n, d, 1)
        cal_agent = generateAgent(1, n, d, 1)
        predictor0 = Predictor(d, hidden_dims_pred)
        predictor0.trainFromAgent(predTrAgent0, epochs=200, mute=True)
        engTrAgent0.loadPred(predictor0)
        cal_agent.loadPred(predictor0)

        setseed(seed_rep)
        gamma = sampleGamma(agentN, cor_ratio)
        predTrAgent = [generateAgent(ga, n, d, 1.5) for ga in gamma]
        engTrAgent = [generateAgent(ga, 2*n, d, 1.5) for ga in gamma]
        predictorList = [Predictor(d, hidden_dims_pred) for _ in predTrAgent]
        for i in range(len(predictorList)):
            predictorList[i].trainFromAgent(predTrAgent[i], epochs=200, mute=True)
            engTrAgent[i].loadPred(predictorList[i])

        setseed(seed_rep)
        for i in range(len(engTrAgent)):
            engTrAgent[i].gamma = gamma[i]
        fclus = FClus(engTrAgent0, engTrAgent, 10, 0)
        gammas = np.array([a.gamma for a in fclus.aux_agentList])
        res_matrix[rep, :] = gammas
        log(', '.join([f"{x:.3f}" for x in gammas]), path=Lpath, islog=isLog)
        ranks = rankdata(gammas, method='max')[::-1]
        log(f"Rank correlate coefficient {np.corrcoef(ranks, range(len(ranks)))[0,1]:.3f}", path=Lpath, islog=isLog)

        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"End at time {tn}", path=Lpath, islog=isLog)

    log("#"*20, path=Lpath, islog=isLog)
    np.save(Rpath+f"/res_matrix_{cor_ratio}.npy", res_matrix)
    ranks_coef = np.array([np.corrcoef(rankdata(res_matrix[i, :], method='max')[::-1], range(agentN))[0,1] for i in range(repeats)])
    ranks_coef = ranks_coef[~np.isnan(ranks_coef)]
    mean_corr = np.mean(ranks_coef)
    log(f"Rank correlate coefficient mean: {mean_corr:.3f}", path=Lpath, islog=isLog)