import copy
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
from FCPCl import *
from FedCPCl import *
from FedCPQQCl import *
from CPhetCl import *
from CPlabCl import *
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

def predict_sum(fcp, test_agent_, test_agent, name, alpha, isLog, Lpath, log):
    indX = np.int_(test_agent_.getY().squeeze())
    cs = fcp.predict(test_agent.getX())
    cov = cs[np.arange(cs.shape[0]), np.int_(test_agent.getY()).squeeze()]
    size = cs.sum(-1)
    mar0, size0, local_cov0 = one_summation_real(cov, size, indX, alpha)
    logResult_(mar0, size0, local_cov0, name, isLog=isLog, path=Lpath, log=log)
    return cs, cov, size, indX

def toInd(indX, X_ind_unique):
    return (indX.reshape((-1,1))>=X_ind_unique.reshape((1,-1))).sum(-1)

def auto_skew_transform(df, log, path, isLog, skew_threshold=1.0):
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
    log(f"Applied log1p to "+', '.join([f"{col_list[i]} ({skewness_list[i]:.2f})" for i in range(len(col_list))]), path, islog=isLog)
    return df, skewed_features

def kernel_weight(y, loc, scale):
    w = np.exp(-(y-loc)**2/(2*scale**2))
    w = w / np.sum(w)
    return w

def sampleAux(agent_aux, n):
    """
    划分pred agent和eng agent，前者每个n个样本，后者2n个
    """
    predTrAgent_list = []
    engTrAgent_list = []
    for i in range(len(agent_aux)):
        # N = agent_aux[i].n
        # predTrAgent, engTrAgent = agent_aux[i].splitAgent(N//2)
        agent_, _ = agent_aux[i].splitAgent(3*n)
        predTrAgent, engTrAgent = agent_.splitAgent(n)
        predTrAgent_list.append(predTrAgent)
        engTrAgent_list.append(engTrAgent)
    return predTrAgent_list, engTrAgent_list

def selectTarget(images, labels, tar_N, aux_N):
    images_tar = np.zeros([0]+list(images.shape[1:]))
    images_aux = np.zeros([0]+list(images.shape[1:]))
    labels_tar = np.zeros((0, 1))
    labels_aux = np.zeros((0, 1))
    for i in range(len(tar_N)):
        mask = (labels == i).squeeze()
        X_, Y_ = images[mask], labels[mask]
        inds = np.random.permutation(len(X_))
        images_tar = np.concatenate([images_tar, X_[inds[:tar_N[i]*4]]])
        labels_tar = np.concatenate([labels_tar, Y_[inds[:tar_N[i]*4]]])
        images_aux = np.concatenate([images_aux, X_[inds[tar_N[i]*4:tar_N[i]*4+2*aux_N[i]]]])
        labels_aux = np.concatenate([labels_aux, Y_[inds[tar_N[i]*4:tar_N[i]*4+2*aux_N[i]]]])
    agent_target = Agent(images_tar, labels_tar)
    agent_aux = Agent(images_aux, labels_aux)
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
    if os.path.split(os.getcwd())[1] != 'Real_Derma':
        os.chdir(os.path.join(os.getcwd(), 'Real_Derma'))

    if len(sys.argv) > 1:
        n, auxn, auxnum, w0, isLog = int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]), float(sys.argv[4]), True
    else:
        n, auxn, auxnum, w0, isLog = 200, 250, 8, 0.01, False

    seed = 1
    setseed(seed)
    auxN = auxn * auxnum
    data = np.load('../Dataset/dermamnist.npz')
    images = data['train_images']
    labels = data['train_labels']
    typeNum = 7
    tar_w = [0.15-w0,0.25-w0,0.2-w0,0.05,0.2-w0,0.1+4*w0,0.05]
    tar_N = [int(n*i) for i in tar_w]
    tar_N[-1] = n-np.sum(tar_N)+tar_N[-1]
    aux_N = [min(i, 200) for i in (np.unique(labels, return_counts=True)[1] - 4*np.array(tar_N))//2]
    aux_N[-2] = auxN - np.sum(aux_N) + aux_N[-2]
    agent_target, agent_aux = selectTarget(images, labels, tar_N, aux_N)

    epoches, repeats, testN, d = 200, 100, np.sum(tar_N), 10
    hidden_dims_pred = [30, 30]
    hidden_dims_dre = [30]
    num_layer, hidden_dim, noise_dim = 2, 100, d
    alpha = 0.1
    trainerMap = MNISTTrainer(
        batch_size=64,
        learning_rate=0.001,
        num_epochs=100
    )
    trainerMap.load("Para/DermaMNIST.pth")

    Rpath = f"../Result/Real_Derma/{n}_{auxn}_{auxnum}_{w0}"
    Lpath = f"../Log/Real_Derma/{n}_{auxn}_{auxnum}_{w0}.txt"
    checkDir("../Result")
    checkDir("../Result/Real_Derma")
    checkDir(Rpath)
    checkDir("../Log")
    checkDir("../Log/Real_Derma")

    if isLog:
        relog(Lpath)
        dic = {'n': n, 'auxn':auxn, 'auxnum': auxnum, 'tar_w':tar_w, 'tar_N':tar_N, 'aux_N':np.array(aux_N), 'alpha': alpha, 'seed':seed, 'epoches': epoches, 'repeats': repeats, 'testN': testN, 'hidden_dims_pred': hidden_dims_pred, 'hidden_dims_dre': hidden_dims_dre, "num_layer": num_layer, "hidden_dim": hidden_dim, "noise_dim": noise_dim}
        logInfo(dic, Lpath)
    log(f"Target data pool with {agent_target.n} samples, {auxnum} auxiliary agents each with {agent_aux.n // auxnum} samples. Test sample size is {testN}", path=Lpath, islog=isLog)

    CS0, COV0, SIZE0 = generate_Empty_Cl(repeats, testN, typeNum)
    CS1, COV1, SIZE1 = generate_Empty_Cl(repeats, testN, typeNum)
    CS2, COV2, SIZE2 = generate_Empty_Cl(repeats, testN, typeNum)
    CS3, COV3, SIZE3 = generate_Empty_Cl(repeats, testN, typeNum)
    CS4, COV4, SIZE4 = generate_Empty_Cl(repeats, testN, typeNum)
    CS5, COV5, SIZE5 = generate_Empty_Cl(repeats, testN, typeNum)
    IND = np.zeros((repeats, testN), dtype=int)

    for rep in range(repeats):
        seed_rep = (rep + 1) * (np.abs(seed) + 1)
        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"#"*20+f" Repeat {rep+1} "+f"#"*20, path=Lpath, islog=isLog)
        log(f"Start at time {tn}", path=Lpath, islog=isLog)

        setseed(seed_rep)
        agent_target, agent_aux = selectTarget(images, labels, tar_N, aux_N)
        setseed(seed_rep)
        agent_test_, agent_tar = agent_target.splitAgent(testN)
        agent_tar_predTr, agent_tar = agent_tar.splitAgent(n)
        agent_tar_engTr, cal_agent_ = agent_tar.splitAgent(n)
        predTrAgent, engTrAgent_ = [], []
        agent_a = copy.deepcopy(agent_aux)
        for i in range(auxnum):
            pa, agent_a = agent_a.splitAgent(agent_aux.n // (2*auxnum))
            predTrAgent.append(pa)
            ea, agent_a = agent_a.splitAgent(agent_aux.n // (2*auxnum))
            engTrAgent_.append(ea)
        predTrAgent = [agent_tar_predTr] + predTrAgent
        engTrAgent_ = [agent_tar_engTr] + engTrAgent_

        setseed(seed_rep)
        predictorList = [MNISTTrainer(batch_size=64, learning_rate=0.001, num_epochs=30) for _ in predTrAgent]
        for i in range(len(predictorList)):
            predictorList[i].load("Para/DermaMNIST.pth")
            predictorList[i].prepare_data(train_images=predTrAgent[i].X, train_labels=predTrAgent[i].Y)
            predictorList[i].run_training_classifier2(mute=True)

        setseed(seed_rep)
        engTrAgent = []
        for i in range(len(engTrAgent_)):
            ea = Agent(predictorList[i].covx(engTrAgent_[i].X), engTrAgent_[i].Y)
            ea.loadS(calScoreCl(predictorList[i], engTrAgent_[i].X, engTrAgent_[i].Y))
            engTrAgent.append(ea)
        setseed(seed_rep)
        cal_agent = Agent(predictorList[0].covx(cal_agent_.X), cal_agent_.Y)
        cal_agent.loadS(calScoreCl(predictorList[0], cal_agent_.X, cal_agent_.Y))
        agent_test = Agent(predictorList[0].covx(agent_test_.X), agent_test_.Y)

        setseed(seed_rep)
        engTrAgent_xy = [Agent(a.X, a.Y) for a in engTrAgent]
        for a in engTrAgent_xy:
            a.loadS(a.Y)
        engTrAgent_y = [Agent(np.zeros((a.Y.shape[0], 0)), a.Y) for a in engTrAgent]
        for a in engTrAgent_y:
            a.loadS(a.Y)

        setseed(seed_rep)
        fcdre = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre.trainOnXS(engTrAgent[0], engTrAgent[1:], epoches=epoches // 4, mute=True, caliN=0.1)
        fcdre_xy = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_xy.trainOnXS(engTrAgent_xy[0], engTrAgent_xy, epoches=epoches, mute=True, caliN=0.1)
        fcdre_y = FederatedCalibratedDRE(0, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=True)
        fcdre_y.trainOnXS(engTrAgent_y[0], engTrAgent_y, epoches=epoches, mute=True, caliN=0.1)

        setseed(seed_rep)
        feng1 = FEng(d, num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim)
        feng2 = FEng(d, num_layer=num_layer, hidden_dim=hidden_dim, noise_dim=noise_dim)
        feng1.train([engTrAgent[0]], epoches=epoches, bs=32, mute=True)
        feng2.train(engTrAgent[1:], epoches=epoches, bs=32, mute=True)

        setseed(seed_rep)
        tar_comb = combineAgents([engTrAgent[0], cal_agent])
        fcp0 = FCPCl(predictorList[0], cal_agent, feng1, [], [], alpha=alpha)
        fcp1 = FCPCl(predictorList[0], cal_agent, feng1, [feng2], [fcdre], alpha=alpha)
        fedcp = FedCPCl(predictorList[0], tar_comb, engTrAgent[1:], alpha=alpha)
        fedcpqq = FedCPQQCl(predictorList[0], tar_comb, engTrAgent[1:], alpha=alpha)
        cphet = CPhetCl(predictorList[0], [cal_agent] + engTrAgent[1:], fcdre_xy, alpha)
        cplab = CPlabCl(predictorList[0], [cal_agent] + engTrAgent[1:], fcdre_y, alpha)

        setseed(seed_rep)
        CS0[rep, :, :], COV0[rep, :], SIZE0[rep, :], IND[rep, :] = predict_sum(fcp0, agent_test_, agent_test, 'LCP', alpha, isLog, Lpath, log)
        CS1[rep, :, :], COV1[rep, :], SIZE1[rep, :], _ = predict_sum(fcp1, agent_test_, agent_test, 'FCP', alpha, isLog, Lpath, log)
        CS2[rep, :, :], COV2[rep, :], SIZE2[rep, :], _ = predict_sum(fedcp, agent_test_, agent_test, 'FedCP', alpha, isLog, Lpath, log)
        CS3[rep, :, :], COV3[rep, :], SIZE3[rep, :], _ = predict_sum(fedcpqq, agent_test_, agent_test, 'FedCP-QQ', alpha, isLog, Lpath, log)
        CS4[rep, :, :], COV4[rep, :], SIZE4[rep, :], _ = predict_sum(cphet, agent_test_, agent_test, 'CPhet', alpha, isLog, Lpath, log)
        CS5[rep, :, :], COV5[rep, :], SIZE5[rep, :], _ = predict_sum(cplab, agent_test_, agent_test, 'CPlab', alpha, isLog, Lpath, log)
        tn = datetime.datetime.now().strftime('%H:%M:%S')
        log(f"End at time {tn}", path=Lpath, islog=isLog)

    saveData(Rpath, CS0, COV0, SIZE0, "LCP")
    saveData(Rpath, CS1, COV1, SIZE1, "FCP")
    saveData(Rpath, CS2, COV2, SIZE2, "FedCP")
    saveData(Rpath, CS3, COV3, SIZE3, "FedCP-QQ")
    saveData(Rpath, CS4, COV4, SIZE4, "CPhet")
    saveData(Rpath, CS5, COV5, SIZE5, "CPlab")
    np.save(f"{Rpath}/Test_Index.npy", IND)
    summation_real(COV0, SIZE0, IND, "LCP", alpha, isLog, Lpath, log)
    summation_real(COV1, SIZE1, IND, "FCP", alpha, isLog, Lpath, log)
    summation_real(COV2, SIZE2, IND, "FedCP", alpha, isLog, Lpath, log)
    summation_real(COV3, SIZE3, IND, "FedCP-QQ", alpha, isLog, Lpath, log)
    summation_real(COV4, SIZE4, IND, "CPhet", alpha, isLog, Lpath, log)
    summation_real(COV5, SIZE5, IND, "CPlab", alpha, isLog, Lpath, log)