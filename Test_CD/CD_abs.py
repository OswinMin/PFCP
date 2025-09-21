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
            y = (y - x.sum(-1).reshape((-1, 1))) / sigma(X, gamma)   # regularized y
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

def sampleGamma(agentN, gammaSelectRule, g:float=.2):
    if gammaSelectRule == 0:
        gamma = np.random.uniform(1-g, 1, agentN)
    elif gammaSelectRule == 1:
        gamma = np.random.uniform(1, 1+2*g, agentN)
    else:
        gamma = np.random.uniform(1-g, 1+g, agentN)
        gamma[gamma>1] = 2 * (gamma[gamma>1] - 1) + 1
    return gamma

def predict_sum(fcp, test_agent, name, isLog, Lpath, log):
    cs = fcp.predict(test_agent.getX())
    cov = test_agent.coverage(test_agent.getX(), cs).reshape(-1)
    size = cs[:, 1] - cs[:, 0]
    mar0, size0, local_cov0 = one_summation(cov, size, .1)
    logResult_(mar0, size0, local_cov0, name, isLog=isLog, path=Lpath, log=log)
    return cs, cov, size

if __name__ == '__main__':
    if os.path.split(os.getcwd())[1] != 'Test_CD':
        os.chdir(os.path.join(os.getcwd(), 'Test_CD'))
    # each agent has 3*n samples
    d, n, agentN, gammaSelectRule, g, seed = 9, 100, 20, 0, .2, 1
    if len(sys.argv) > 1:
        d, n, agentN, gammaSelectRule, g, seed = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), float(sys.argv[5]), int(sys.argv[6])
        isLog = True
    else:
        isLog = False

    Rpath = f"../Result/CD_abs/{d}_{n}_{agentN}_{gammaSelectRule}_{g}"
    Lpath = f"../Log/CD_abs/{d}_{n}_{agentN}_{gammaSelectRule}_{g}.txt"
    checkDir("../Result")
    checkDir("../Result/CD_abs")
    checkDir(Rpath)
    checkDir("../Log")
    checkDir("../Log/CD_abs")

    epoches = 500
    repeats = 100
    testN = 1000
    hidden_dims_pred = [30, 30]
    hidden_dims_dre = []
    hidden_dims_dre_ = [30]
    num_layer, hidden_dim, noise_dim = 2, 100, d
    if isLog:
        relog(Lpath)
        dic = {'d': d, 'n': n, 'agentN': agentN, 'gammaSelectRule': gammaSelectRule, 'g': g, 'seed': seed, 'epoches': epoches, 'repeats': repeats, 'testN': testN, 'hidden_dims_pred': hidden_dims_pred, 'hidden_dims_dre': hidden_dims_dre, "hidden_dims_dre_": hidden_dims_dre_, "num_layer": num_layer, "hidden_dim": hidden_dim, "noise_dim": noise_dim}
        logInfo(dic, Lpath)

    setseed(seed)
    test_agent = generateAgent(1, testN, d, 1)
    CS0, COV0, SIZE0 = generate_Empty(repeats, testN)
    CS1, COV1, SIZE1 = generate_Empty(repeats, testN)
    CS2, COV2, SIZE2 = generate_Empty(repeats, testN)
    CS3, COV3, SIZE3 = generate_Empty(repeats, testN)
    CS4, COV4, SIZE4 = generate_Empty(repeats, testN)
    CS5, COV5, SIZE5 = generate_Empty(repeats, testN)
    CS6, COV6, SIZE6 = generate_Empty(repeats, testN)

    for rep in range(repeats):
        seed_rep = (rep + 1) * (np.abs(seed) + 1)
        setseed(seed_rep)
        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"#"*20+f" Repeat {rep+1} "+f"#"*20, path=Lpath, islog=isLog)
        log(f"Start at time {tn}", path=Lpath, islog=isLog)

        gamma = sampleGamma(agentN, gammaSelectRule, g)
        predTrAgent = [generateAgent(1, n, d, 1)] + [generateAgent(ga, n, d, 1.5) for ga in gamma]
        engTrAgent = [generateAgent(1, n, d, 1)] + [generateAgent(ga, 2*n, d, 1.5) for ga in gamma]
        cal_agent = generateAgent(1, n, d, 1)

        predictorList = [Predictor(d, hidden_dims_pred) for _ in predTrAgent]
        for i in range(len(predictorList)):
            predictorList[i].trainFromAgent(predTrAgent[i], epochs=200, mute=True)
            engTrAgent[i].loadPred(predictorList[i])
        cal_agent.loadPred(predictorList[0])

        predTrAgent_xy = [Agent(a.X, a.Y) for a in predTrAgent]
        for a in predTrAgent_xy:
            a.loadS(a.Y)
        predTrAgent_y = [Agent(np.zeros((a.Y.shape[0], 0)), a.Y) for a in predTrAgent]
        for a in predTrAgent_y:
            a.loadS(a.Y)

        setseed(seed_rep)
        fcdre = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=False)
        fcdre.trainOnXS(predTrAgent[0], predTrAgent[1:], epoches=epoches, mute=True)
        fcdre_ = FederatedCalibratedDRE(d, hidden_dims_dre_, lr=5e-3, onXS=True, calibrate=True)
        fcdre_.trainOnXS(predTrAgent[0], predTrAgent[1:], epoches=epoches, mute=True, caliN=0.1)
        fcdre_xy = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_xy.trainOnXS(predTrAgent_xy[0], predTrAgent_xy, epoches=epoches//2, mute=True, caliN=0.1)
        fcdre_y = FederatedCalibratedDRE(0, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_y.trainOnXS(predTrAgent_y[0], predTrAgent_y, epoches=epoches//2, mute=True, caliN=0.1)

        setseed(seed_rep)
        feng1 = FEng(d, num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim)
        feng2 = FEng(d, num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim)
        feng1.train([engTrAgent[0]], epoches=epoches, bs=32, mute=True)
        feng2.train(engTrAgent[1:], epoches=epoches, bs=32, mute=True)

        setseed(seed_rep)
        tar_comb = combineAgents([engTrAgent[0], cal_agent])
        tar_comb.loadPred(predictorList[0])
        fcp0 = FCP(cal_agent, feng1, [], [])
        fcp1 = FCP(cal_agent, feng1, [feng2], [fcdre])
        fcp2 = FCP(cal_agent, feng1, [feng2], [fcdre_])
        fedcp = FedCP(tar_comb, engTrAgent[1:], .1)
        fedcpqq = FedCPQQ(tar_comb, engTrAgent[1:], .1)
        cphet = CPhet(cal_agent, [cal_agent]+engTrAgent[1:], fcdre_xy, .1)
        cplab = CPlab(cal_agent, [cal_agent]+engTrAgent[1:], fcdre_y, .1)

        CS0[rep, :, :], COV0[rep, :], SIZE0[rep, :] = predict_sum(fcp0, test_agent, 'LCP', isLog, Lpath, log)
        CS1[rep, :, :], COV1[rep, :], SIZE1[rep, :] = predict_sum(fcp1, test_agent, 'FCP', isLog, Lpath, log)
        CS2[rep, :, :], COV2[rep, :], SIZE2[rep, :] = predict_sum(fcp2, test_agent, 'FCP_', isLog, Lpath, log)
        CS3[rep, :, :], COV3[rep, :], SIZE3[rep, :] = predict_sum(fedcp, test_agent, 'FedCP', isLog, Lpath, log)
        CS4[rep, :, :], COV4[rep, :], SIZE4[rep, :] = predict_sum(fedcpqq, test_agent, 'FedCP-QQ', isLog, Lpath, log)
        CS5[rep, :, :], COV5[rep, :], SIZE5[rep, :] = predict_sum(cphet, test_agent, 'CPhet', isLog, Lpath, log)
        CS6[rep, :, :], COV6[rep, :], SIZE6[rep, :] = predict_sum(cplab, test_agent, 'CPlab', isLog, Lpath, log)

        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"End at time {tn}", path=Lpath, islog=isLog)

    mar0, size0, local_cov0, tt_local_cov0 = summation(COV0, SIZE0, .1)
    mar1, size1, local_cov1, tt_local_cov1 = summation(COV1, SIZE1, .1)
    mar2, size2, local_cov2, tt_local_cov2 = summation(COV2, SIZE2, .1)
    mar3, size3, local_cov3, tt_local_cov3 = summation(COV3, SIZE3, .1)
    mar4, size4, local_cov4, tt_local_cov4 = summation(COV4, SIZE4, .1)
    mar5, size5, local_cov5, tt_local_cov5 = summation(COV5, SIZE5, .1)
    mar6, size6, local_cov6, tt_local_cov6 = summation(COV6, SIZE6, .1)
    saveData(Rpath, CS0, COV0, SIZE0, "LCP")
    saveData(Rpath, CS1, COV1, SIZE1, "FCP")
    saveData(Rpath, CS2, COV2, SIZE2, "FCP_")
    saveData(Rpath, CS3, COV3, SIZE3, "FedCP")
    saveData(Rpath, CS4, COV4, SIZE4, "FedCP-QQ")
    saveData(Rpath, CS5, COV5, SIZE5, "CPhet")
    saveData(Rpath, CS6, COV6, SIZE6, "CPlab")
    logResult(mar0, size0, local_cov0, tt_local_cov0, "LCP", isLog=isLog, path=Lpath, log=log)
    logResult(mar1, size1, local_cov1, tt_local_cov1, "FCP", isLog=isLog, path=Lpath, log=log)
    logResult(mar2, size2, local_cov2, tt_local_cov2, "FCP_", isLog=isLog, path=Lpath, log=log)
    logResult(mar3, size3, local_cov3, tt_local_cov3, "FedCP", isLog=isLog, path=Lpath, log=log)
    logResult(mar4, size4, local_cov4, tt_local_cov4, "FedCP-QQ", isLog=isLog, path=Lpath, log=log)
    logResult(mar5, size5, local_cov5, tt_local_cov5, "CPhet", isLog=isLog, path=Lpath, log=log)
    logResult(mar6, size6, local_cov6, tt_local_cov6, "CPlab", isLog=isLog, path=Lpath, log=log)
