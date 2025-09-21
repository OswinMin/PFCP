from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Union
from Agent import *
from tools import *
from Predictor import *
from FDRE import *
from FEng import *

class FCP:
    def __init__(self, targetAgent:Agent, feng:FEng, fengList:list[FEng]=[], fdreList:list[FederatedCalibratedDRE]=[], alpha:float=.1, n:int=1000):
        super().__init__()
        self.targetAgent = targetAgent      # calibration dataset
        self.feng = feng
        self.predictor = self.targetAgent.pred
        self.alpha = alpha
        self.fengList = fengList
        self.fdreList = fdreList
        self.hasf = (len(self.fengList) > 0) and (len(self.fdreList) > 0)
        self.center = self.targetAgent.getCenter()
        self.quantileOnCalibration(n)

    def quantileOnCalibration(self, n:int=1000):
        self.n = n
        self.maxS = np.max(self.targetAgent.getS())
        w = self.feng.calW(self.targetAgent.getX(), self.center, 1 / (self.targetAgent.n + 1))
        if self.hasf:
            self.beta = self.feng.percentileCombineOther(self.targetAgent.getX(), self.targetAgent.getS(), n, self.fengList, self.fdreList, maxS=self.maxS, w=w)
        else:
            self.beta = self.feng.percentile(self.targetAgent.getX(), self.targetAgent.getS(), n, maxS=self.maxS, w=w)
        self.betas = np.concatenate([self.beta, [1.]])
        self.q = np.quantile(self.betas, 1-self.alpha)

    def predict(self, testX):
        yhat = self.predictor.predict(testX)
        w = self.feng.calW(testX, self.center, 1 / (self.targetAgent.n + 1))
        self.conformalSet = np.ones((yhat.shape[0], 2)) * yhat.reshape((-1,1))
        if self.hasf:
            sq = self.feng.getQuantileCombineOther(testX, self.fengList, self.fdreList, self.q, self.n, maxS=self.maxS, w=w)
        else:
            sq = self.feng.getQuantile(testX, self.q, self.n, maxS=self.maxS, w=w)
        # sq_ = np.clip(sq, 0, np.max(sq)).reshape(-1)
        sq_ = np.clip(sq, 0, self.maxS*2).reshape(-1)
        self.conformalSet[:, 0] = self.conformalSet[:, 0] - sq_
        self.conformalSet[:, 1] = self.conformalSet[:, 1] + sq_
        return self.conformalSet

if __name__ == '__main__':
    def generateXY(gamma, n, d):
        X = np.random.normal(0, 1, (n, d))
        Y = X.sum(-1).reshape((-1, 1)) + np.random.normal(0, 1, (n, 1)) * np.cos(X.sum(-1)).reshape((-1, 1)) * gamma
        return X, Y


    def generateRepXY(gamma, n, d, m):
        X = np.random.normal(0, 1, (n, d))
        Y = X.sum(-1).reshape((-1, 1)) + np.random.normal(0, 1, (n, m)) * np.cos(X.sum(-1)).reshape((-1, 1)) * gamma
        return X, Y


    setseed(0)
    n = 100
    hidden_dims_pred = [30, 30]
    hidden_dims_dre = []
    d = 5
    epoches = 500

    gamma = np.random.uniform(0.9, 1, 20)
    predTrAgent = [Agent(*generateXY(1, n, d))] + [Agent(*generateXY(ga, n, d)) for ga in gamma]
    engTrAgent = [Agent(*generateXY(1, n, d))] + [Agent(*generateXY(ga, 2 * n, d)) for ga in gamma]
    calagent = Agent(*generateXY(1, n, d))

    predictorList = [Predictor(d, hidden_dims_pred) for _ in predTrAgent]
    for i in range(len(predictorList)):
        predictorList[i].trainFromAgent(predTrAgent[i], epochs=200, mute=True)
        engTrAgent[i].loadPred(predictorList[i])
    calagent.loadPred(predictorList[0])

    X, Y = generateRepXY(1, 200, d, 1000)

    fcdre = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=False)
    fcdre.trainOnXS(engTrAgent[0], engTrAgent[1:], caliN=5, epoches=epoches, log=log, mute=False)

    feng1 = FEng(d)
    feng1.train([engTrAgent[0]], epoches=300, bs=32, mute=False)

    feng2 = FEng(d)
    feng2.train(engTrAgent[1:], epoches=300, bs=32, mute=False)

    for k in [1,2,3]:
        setseed(0)
        fcp1 = FCP(calagent, feng1, [], [])
        fcp2 = FCP(calagent, feng1, [feng2]*k, [fcdre]*k)

        cs1 = fcp1.predict(X)
        cs2 = fcp2.predict(X)
        size1 = (cs1[:,1] - cs1[:,0]).mean()
        size2 = (cs2[:,1] - cs2[:,0]).mean()
        loc_cov1 = ((cs1[:,[0]]<=Y)&(cs1[:,[1]]>=Y)).mean(-1)
        loc_cov2 = ((cs2[:,[0]]<=Y)&(cs2[:,[1]]>=Y)).mean(-1)
        mar1 = np.mean(loc_cov1)
        mar2 = np.mean(loc_cov2)
        cta1 = np.mean(np.abs(loc_cov1-.9))
        cta2 = np.mean(np.abs(loc_cov2-.9))
        print(f"LCP: mar={mar1:.4f}, cta={cta1:.4f}, size={size1:.4f}")
        print(f"FCP {k}: mar={mar2:.4f}, cta={cta2:.4f}, size={size2:.4f}")
