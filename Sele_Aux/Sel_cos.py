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
from Agent import *
from tools import *
from Predictor import *
from FDRE import *
from FEng import *
from FCP import *
from FedCP import *
from FedCPQQ import *
from CPhet import *
from CPlab import *
from FClus import *
import warnings
import datetime
warnings.filterwarnings('ignore')

def sigma(X, gamma):
    return np.abs(np.cos(X.sum(-1))).reshape((-1, 1)) * np.sqrt(gamma)

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
    d, n, agentN, seleagentN, cor_ratio, seed = 20, 100, 40, 20, 0.5, 1
    if len(sys.argv) > 1:
        d, n, agentN, seleagentN, cor_ratio, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
        isLog = True
    else:
        isLog = False

    Rpath = f"../Result/Sele_cos/{d}_{n}_{agentN}_{seleagentN}_{cor_ratio}"
    Lpath = f"../Log/Sele_cos/{d}_{n}_{agentN}_{seleagentN}_{cor_ratio}.txt"
    checkDir("../Result")
    checkDir("../Result/Sele_cos")
    checkDir(Rpath)
    checkDir("../Log")
    checkDir("../Log/Sele_cos")

    epoches = 300
    repeats = 100
    testN = 1000
    hidden_dims_pred = [30, 30]
    hidden_dims_dre = [30]
    num_layer, hidden_dim, noise_dim = 2, 100, d
    if isLog:
        relog(Lpath)
        dic = {'d':d, 'n':n, 'agentN':agentN, 'seleagentN':seleagentN, 'cor_ratio':cor_ratio, 'seed':seed, 'epoches':epoches, 'repeats':repeats, 'testN':testN, 'hidden_dims_pred':hidden_dims_pred, 'hidden_dims_dre':hidden_dims_dre, "num_layer":num_layer, "hidden_dim":hidden_dim, "noise_dim":noise_dim}
        logInfo(dic, Lpath)

    setseed(seed)
    test_agent = generateAgent(1, testN, d, 1)
    CS0, COV0, SIZE0 = generate_Empty(repeats, testN)
    CS1, COV1, SIZE1 = generate_Empty(repeats, testN)
    CS2, COV2, SIZE2 = generate_Empty(repeats, testN)
    CS3, COV3, SIZE3 = generate_Empty(repeats, testN)
    CS4, COV4, SIZE4 = generate_Empty(repeats, testN)
    CS5, COV5, SIZE5 = generate_Empty(repeats, testN)

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

        predTrAgent_xy = [Agent(a.X, a.Y) for a in predTrAgent]
        for a in predTrAgent_xy:
            a.loadS(a.Y)
        predTrAgent_y = [Agent(np.zeros((a.Y.shape[0], 0)), a.Y) for a in predTrAgent]
        for a in predTrAgent_y:
            a.loadS(a.Y)

        setseed(seed_rep)
        fclus = FClus(engTrAgent0, engTrAgent, 10, 0)
        engTrAgent = [engTrAgent0] + fclus.getKImportant(seleagentN)

        setseed(seed_rep)
        fcdre = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre.trainOnXS(engTrAgent[0], engTrAgent[1:], epoches=epoches, mute=True)
        fcdre_xy = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_xy.trainOnXS(predTrAgent_xy[0], predTrAgent_xy, epoches=epoches // 2, mute=True, caliN=0.1)
        fcdre_y = FederatedCalibratedDRE(0, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_y.trainOnXS(predTrAgent_y[0], predTrAgent_y, epoches=epoches // 2, mute=True, caliN=0.1)

        setseed(seed_rep)
        feng1 = FEng(d, num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim)
        feng2 = FEng(d, num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim)
        feng1.train([engTrAgent[0]], epoches=epoches, bs=32, mute=True)
        feng2.train(engTrAgent[1:], epoches=epoches, bs=32, mute=True)

        setseed(seed_rep)
        tar_comb = combineAgents([engTrAgent[0], cal_agent])
        tar_comb.loadPred(predictor0)
        fcp0 = FCP(cal_agent, feng1, [], [])
        fcp1 = FCP(cal_agent, feng1, [feng2], [fcdre])
        fedcp = FedCP(tar_comb, engTrAgent[1:], .1)
        fedcpqq = FedCPQQ(tar_comb, engTrAgent[1:], .1)
        cphet = CPhet(cal_agent, [cal_agent] + engTrAgent[1:], fcdre_xy, .1)
        cplab = CPlab(cal_agent, [cal_agent] + engTrAgent[1:], fcdre_y, .1)

        setseed(seed_rep)
        CS0[rep, :, :], COV0[rep, :], SIZE0[rep, :] = predict_sum(fcp0, test_agent, 'LCP', isLog, Lpath, log)
        CS1[rep, :, :], COV1[rep, :], SIZE1[rep, :] = predict_sum(fcp1, test_agent, 'FCP', isLog, Lpath, log)
        CS2[rep, :, :], COV2[rep, :], SIZE2[rep, :] = predict_sum(fedcp, test_agent, 'FedCP', isLog, Lpath, log)
        CS3[rep, :, :], COV3[rep, :], SIZE3[rep, :] = predict_sum(fedcpqq, test_agent, 'FedCP-QQ', isLog, Lpath, log)
        CS4[rep, :, :], COV4[rep, :], SIZE4[rep, :] = predict_sum(cphet, test_agent, 'CPhet', isLog, Lpath, log)
        CS5[rep, :, :], COV5[rep, :], SIZE5[rep, :] = predict_sum(cplab, test_agent, 'CPlab', isLog, Lpath, log)

        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"End at time {tn}", path=Lpath, islog=isLog)

    mar0, size0, local_cov0, tt_local_cov0 = summation(COV0, SIZE0, .1)
    mar1, size1, local_cov1, tt_local_cov1 = summation(COV1, SIZE1, .1)
    mar2, size2, local_cov2, tt_local_cov2 = summation(COV2, SIZE2, .1)
    mar3, size3, local_cov3, tt_local_cov3 = summation(COV3, SIZE3, .1)
    mar4, size4, local_cov4, tt_local_cov4 = summation(COV4, SIZE4, .1)
    mar5, size5, local_cov5, tt_local_cov5 = summation(COV5, SIZE5, .1)
    saveData(Rpath, CS0, COV0, SIZE0, "LCP")
    saveData(Rpath, CS1, COV1, SIZE1, "FCP")
    saveData(Rpath, CS2, COV2, SIZE2, "FedCP")
    saveData(Rpath, CS3, COV3, SIZE3, "FedCP-QQ")
    saveData(Rpath, CS4, COV4, SIZE4, "CPhet")
    saveData(Rpath, CS5, COV5, SIZE5, "CPlab")
    logResult(mar0, size0, local_cov0, tt_local_cov0, "LCP", isLog=isLog, path=Lpath, log=log)
    logResult(mar1, size1, local_cov1, tt_local_cov1, "FCP", isLog=isLog, path=Lpath, log=log)
    logResult(mar2, size2, local_cov2, tt_local_cov2, "FedCP", isLog=isLog, path=Lpath, log=log)
    logResult(mar3, size3, local_cov3, tt_local_cov3, "FedCP-QQ", isLog=isLog, path=Lpath, log=log)
    logResult(mar4, size4, local_cov4, tt_local_cov4, "CPhet", isLog=isLog, path=Lpath, log=log)
    logResult(mar5, size5, local_cov5, tt_local_cov5, "CPlab", isLog=isLog, path=Lpath, log=log)