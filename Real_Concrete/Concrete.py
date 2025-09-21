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
from sklearn.preprocessing import StandardScaler
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
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import warnings
import datetime
warnings.filterwarnings('ignore')

def summation_real(cov, size, ind, name, alpha=.1, isLog=False, path='', log=log):
    mar = np.mean(cov)
    mean_size = np.mean(size)
    unique_ind = np.unique(ind)
    loc_cov = np.zeros(len(unique_ind))
    weight = np.zeros(len(unique_ind))
    for i in range(len(unique_ind)):
        loc_cov[i] = np.mean(cov[ind == unique_ind[i]])
        weight[i] = np.mean(ind == unique_ind[i])
    loc_err = (weight * np.abs(loc_cov - (1-alpha))).sum()
    logResult(mar, mean_size, loc_err, loc_cov, weight, alpha, name, isLog, path, log)

def logResult(mar, size, loc_err, loc_cov, weight, alpha, name, isLog=False, path='', log=log):
    log("#"*20, path, isLog)
    log(f"{name.upper()} : marginal : {mar:.4f}, mean size : {size:.4f}, conditional cov error : {loc_err:.4f}", path=path, islog=isLog)
    log(f"Conditional coverage error and selection weight on each bin:", path=path, islog=isLog)
    log("  ".join([f"{lcc-(1-alpha):.3f}" for lcc in loc_cov]), path=path, islog=isLog)
    # log("  ".join([f"{w:.3f}" for w in weight]), path=path, islog=isLog)

def logResult_(mar, size, local_cov, name, isLog=False, path='', log=log):
    log(f"{name.upper()} : marginal : {mar:.4f}, mean size : {size:.4f}, CTA error : {local_cov:.4f}", path=path, islog=isLog)

def logInfo(dic, Lpath):
    for k, v in dic.items():
        log(f"{k} : {v}", path=Lpath, islog=True)

def one_summation_real(cov, size, indX, alpha=.1):
    mar0 = np.mean(cov)
    size0 = np.mean(size)
    loc_err0 = 0
    for i in np.unique(indX):
        loc_err0 += np.abs((cov[indX == i]-(1-alpha)).sum())
    loc_err0 /= len(indX)
    return mar0, size0, loc_err0

def predict_sum(fcp, test_agent, km, name, alpha, isLog, Lpath, log):
    indX = km.predict(test_agent.getX())
    cs = fcp.predict(test_agent.getX())
    cov = (cs[:, 0]<=test_agent.getY().reshape(-1)) & (cs[:, 1]>=test_agent.getY().reshape(-1))
    size = cs[:, 1] - cs[:, 0]
    mar0, size0, local_cov0 = one_summation_real(cov, size, indX, alpha)
    logResult_(mar0, size0, local_cov0, name, isLog=isLog, path=Lpath, log=log)
    return cs, cov, size, indX

def toInd(indX, X_ind_unique):
    return (indX.reshape((-1,1))>=X_ind_unique.reshape((1,-1))).sum(-1)

def auto_skew_transform(df, skew_threshold=1.0):
    skewed_features = []
    col_list = []
    skewness_list = []
    for col in df.columns:
        skewness = df[col].skew()
        if abs(skewness) > skew_threshold:
            skewed_features.append(col)
            df[col] = np.log1p(df[col])
            col_list.append(col)
            skewness_list.append(skewness)
    return df, skewed_features

def kernel_weight(y, loc, scale):
    w = np.exp(-(y-loc)**2/(2*scale**2))
    w = w / np.sum(w)
    return w

def sampleAux(agent_aux, n, auxN, auxType=0):
    """
    划分pred agent和eng agent，前者每个n个样本，后者2n个
    """
    predTrAgent_list = []
    engTrAgent_list = []
    for i in range(auxN):
        if auxType == 0:
            predTrAgent, agent_aux = agent_aux.splitAgent(n)
            engTrAgent, agent_aux = agent_aux.splitAgent(2*n)
        else:
            predTrAgent, agent_aux = agent_aux.splitAgent(int(1.5*n))
            engTrAgent, agent_aux = agent_aux.splitAgent(3*n-int(1.5*n))
        predTrAgent_list.append(predTrAgent)
        engTrAgent_list.append(engTrAgent)
    return predTrAgent_list, engTrAgent_list

def selectTarget(X, y, mask):
    agent_target = Agent(X[mask], y[mask])
    agent_aux = Agent(X[~mask], y[~mask])
    return agent_target, agent_aux

def splitX(X, k):
    KM = KMeans(n_clusters=k, random_state=42)
    KM.fit(X)
    return KM

def splitData(X, y, ind):
    """
    根据ind指标划分出两组data
    :return:
    """
    mask = np.zeros(len(y), dtype=bool)
    mask[ind] = True
    X1, y1 = X[mask], y[mask]
    X2, y2 = X[~mask], y[~mask]
    return X1, y1, X2, y2

def selectN(y, loc, scale, n):
    w = kernel_weight(y, loc, scale)
    full_ind = np.array(range(len(y)))
    ind = np.random.choice(full_ind, size=n, replace=True, p=w)
    ind = np.unique(ind)
    mask = np.zeros(len(y), dtype=bool)
    mask[ind] = True
    while (len(ind) < n):
        w_ = w[~mask]
        w_ = w_ / np.sum(w_)
        ind_ = np.random.choice(full_ind[~mask], size=n-len(ind), replace=True, p=w_)
        ind = np.unique(np.concatenate((ind, ind_)))
    return ind

if __name__ == '__main__':
    """
    Split agent using school location
    Target: 3*3 choices (g1,2,3 school location rural, suburban or (inner city or urban))
    Rest data serve as auxiliary agent
    use all covariates by kmeans to split the support of X to calculate conditional coverage
    """
    if os.path.split(os.getcwd())[1] != 'Real_Concrete':
        os.chdir(os.path.join(os.getcwd(), 'Real_Concrete'))
    concretedata = pd.read_csv("../Dataset/concrete.csv")
    X_raw = concretedata.drop('csMPa', axis=1)
    y = concretedata['csMPa'].values
    d = X_raw.shape[1]
    # each agent has 3*n samples
    # auxType = 0 : pred tr : eng tr = 1 : 2, otherwise 1 : 1
    n, Y_int, tar_ind, sepN, seed, nauxN, alpha = 50, 10, 3, 10, 1, 0, .2
    if len(sys.argv) > 1:
        n, Y_int, tar_ind, sepN, seed, nauxN, alpha = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), int(sys.argv[6]), float(sys.argv[7])
        isLog = True
    else:
        isLog = False

    auxType, epoches, repeats, testN = 0, 300, 100, 2 * n
    hidden_dims_pred = [30, 30]
    hidden_dims_dre = [30]
    num_layer, hidden_dim, noise_dim = 2, 100, d

    setseed(seed)
    X_transformed, skewed_cols = auto_skew_transform(X_raw.copy(), skew_threshold=1.0)
    scaler = StandardScaler()
    X = pd.DataFrame(scaler.fit_transform(X_transformed), columns=X_raw.columns).values
    nLoc = [i / Y_int for i in range(Y_int + 1)]
    y_loc = np.quantile(y, nLoc)
    y_scale = [(y_loc[i + 1] - y_loc[i - 1]) / 1 for i in range(1, 10)]
    y_scale = [y_scale[0]] + y_scale + [y_scale[-1]]

    Xt, yt, X, y = splitData(X, y, selectN(y, y_loc[tar_ind], y_scale[tar_ind], testN + 3 * n))
    agent_target, agent_aux = Agent(Xt, yt), Agent(X, y)
    km = splitX(Xt, sepN)
    auxN = agent_aux.n // (3*n) - nauxN

    Rpath = f"../Result/Real_Concrete/{n}_{auxN}_{tar_ind}_{sepN}_{seed}_{alpha}"
    Lpath = f"../Log/Real_Concrete/{n}_{auxN}_{tar_ind}_{sepN}_{seed}_{alpha}.txt"
    checkDir("../Result")
    checkDir("../Result/Real_Concrete")
    checkDir(Rpath)
    checkDir("../Log")
    checkDir("../Log/Real_Concrete")

    if isLog:
        relog(Lpath)
        dic = {'d':d, 'n':n, 'Y_int':Y_int, 'tar_ind':tar_ind, 'sepN':sepN, 'seed':seed, 'auxType': auxType, 'alpha':alpha, 'epoches':epoches, 'repeats':repeats, 'testN':testN, 'hidden_dims_pred':hidden_dims_pred, 'hidden_dims_dre':hidden_dims_dre, "num_layer":num_layer, "hidden_dim":hidden_dim, "noise_dim":noise_dim}
        logInfo(dic, Lpath)
    log(f"Target data pool with {agent_target.n} samples, auxiliary agent with {agent_aux.n} samples. In total {auxN} auxiliary agents. Test sample size is {testN}", path=Lpath, islog=isLog)

    CS0, COV0, SIZE0 = generate_Empty(repeats, testN)
    CS1, COV1, SIZE1 = generate_Empty(repeats, testN)
    CS2, COV2, SIZE2 = generate_Empty(repeats, testN)
    CS3, COV3, SIZE3 = generate_Empty(repeats, testN)
    CS4, COV4, SIZE4 = generate_Empty(repeats, testN)
    CS5, COV5, SIZE5 = generate_Empty(repeats, testN)
    CS6, COV6, SIZE6 = generate_Empty(repeats, testN)
    IND = np.zeros((repeats, testN), dtype=int)

    for rep in range(repeats):
        seed_rep = (rep + 1) * (np.abs(seed) + 1)
        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"#"*20+f" Repeat {rep+1} "+f"#"*20, path=Lpath, islog=isLog)
        log(f"Start at time {tn}", path=Lpath, islog=isLog)

        setseed(seed_rep)
        agent_test, agent_tar = agent_target.splitAgent(testN)
        agent_tar_predTr, agent_tar = agent_tar.splitAgent(n)
        agent_tar_engTr, cal_agent = agent_tar.splitAgent(n)
        predTrAgent, engTrAgent = sampleAux(agent_aux, n, auxN, auxType)
        predTrAgent = [agent_tar_predTr] + predTrAgent
        engTrAgent = [agent_tar_engTr] + engTrAgent

        setseed(seed_rep)
        predictorList = [Predictor(d, hidden_dims_pred) for _ in predTrAgent]
        setseed(seed_rep)
        for i in range(len(predictorList)):
            predictorList[i].trainFromAgent(predTrAgent[i], epochs=200, mute=True)
            engTrAgent[i].loadPred(predictorList[i])
        cal_agent.loadPred(predictorList[0])

        engTrAgent_xy = [Agent(a.X, a.Y) for a in engTrAgent]
        for a in engTrAgent_xy:
            a.loadS(a.Y)
        engTrAgent_y = [Agent(np.zeros((a.Y.shape[0], 0)), a.Y) for a in engTrAgent]
        for a in engTrAgent_y:
            a.loadS(a.Y)

        setseed(seed_rep)
        fcdre = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre.trainOnXS(predTrAgent[0], predTrAgent[1:], epoches=epoches, mute=True, caliN=0.1)
        fcdre_ = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_.trainOnXS(engTrAgent[0], engTrAgent[1:], epoches=epoches, mute=True, caliN=0.1)
        fcdre_xy = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_xy.trainOnXS(engTrAgent_xy[0], engTrAgent_xy, epoches=epoches // 2, mute=True, caliN=0.1)
        fcdre_y = FederatedCalibratedDRE(0, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_y.trainOnXS(engTrAgent_y[0], engTrAgent_y, epoches=epoches // 2, mute=True, caliN=0.1)

        setseed(seed_rep)
        feng1 = FEng(d, num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim)
        feng2 = FEng(d, num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim)
        feng1.train([engTrAgent[0]], epoches=epoches, bs=32, mute=True)
        feng2.train(engTrAgent[1:], epoches=epoches, bs=32, mute=True)

        setseed(seed_rep)
        tar_comb = combineAgents([engTrAgent[0], cal_agent])
        tar_comb.loadPred(predictorList[0])
        fcp0 = FCP(cal_agent, feng1, [], [], alpha=alpha)
        fcp1 = FCP(cal_agent, feng1, [feng2], [fcdre], alpha=alpha)
        fcp2 = FCP(cal_agent, feng1, [feng2], [fcdre_], alpha=alpha)
        fedcp = FedCP(tar_comb, engTrAgent[1:], alpha=alpha)
        fedcpqq = FedCPQQ(tar_comb, engTrAgent[1:], alpha=alpha)
        cphet = CPhet(cal_agent, [cal_agent] + engTrAgent[1:], fcdre_xy, alpha)
        cplab = CPlab(cal_agent, [cal_agent] + engTrAgent[1:], fcdre_y, alpha)

        setseed(seed_rep)
        CS0[rep, :, :], COV0[rep, :], SIZE0[rep, :], IND[rep, :] = predict_sum(fcp0, agent_test, km, 'LCP', alpha, isLog, Lpath, log)
        CS1[rep, :, :], COV1[rep, :], SIZE1[rep, :], _ = predict_sum(fcp1, agent_test, km, 'FCP', alpha, isLog, Lpath, log)
        CS2[rep, :, :], COV2[rep, :], SIZE2[rep, :], _ = predict_sum(fcp2, agent_test, km, 'FCP_', alpha, isLog, Lpath, log)
        CS3[rep, :, :], COV3[rep, :], SIZE3[rep, :], _ = predict_sum(fedcp, agent_test, km, 'FedCP', alpha, isLog, Lpath, log)
        CS4[rep, :, :], COV4[rep, :], SIZE4[rep, :], _ = predict_sum(fedcpqq, agent_test, km, 'FedCP-QQ', alpha, isLog, Lpath, log)
        CS5[rep, :, :], COV5[rep, :], SIZE5[rep, :], _ = predict_sum(cphet, agent_test, km, 'CPhet', alpha, isLog, Lpath, log)
        CS6[rep, :, :], COV6[rep, :], SIZE6[rep, :], _ = predict_sum(cplab, agent_test, km, 'CPlab', alpha, isLog, Lpath, log)
        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"End at time {tn}", path=Lpath, islog=isLog)

    saveData(Rpath, CS0, COV0, SIZE0, "LCP")
    saveData(Rpath, CS1, COV1, SIZE1, "FCP")
    saveData(Rpath, CS2, COV2, SIZE2, "FCP_")
    saveData(Rpath, CS3, COV3, SIZE3, "FedCP")
    saveData(Rpath, CS4, COV4, SIZE4, "FedCP-QQ")
    saveData(Rpath, CS5, COV5, SIZE5, "CPhet")
    saveData(Rpath, CS6, COV6, SIZE6, "CPlab")
    np.save(f"{Rpath}/Test_Index.npy", IND)
    summation_real(COV0, SIZE0, IND, "LCP", alpha, isLog, Lpath, log)
    summation_real(COV1, SIZE1, IND, "FCP", alpha, isLog, Lpath, log)
    summation_real(COV2, SIZE2, IND, "FCP_", alpha, isLog, Lpath, log)
    summation_real(COV3, SIZE3, IND, "FedCP", alpha, isLog, Lpath, log)
    summation_real(COV4, SIZE4, IND, "FedCP-QQ", alpha, isLog, Lpath, log)
    summation_real(COV5, SIZE5, IND, "CPhet", alpha, isLog, Lpath, log)
    summation_real(COV6, SIZE6, IND, "CPlab", alpha, isLog, Lpath, log)