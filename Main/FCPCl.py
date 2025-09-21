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
from CNNnet import *

class FCPCl:
    def __init__(self, pred:MNISTTrainer, targetAgent:Agent, feng:FEng, fengList:list[FEng]=[], fdreList:list[FederatedCalibratedDRE]=[], alpha:float=.1, n:int=1000):
        super().__init__()
        self.targetAgent = targetAgent      # calibration dataset
        self.feng = feng
        self.predictor = pred
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
        yhat = 1 - self.predictor.predict(testX)
        w = self.feng.calW(testX, self.center, 1 / (self.targetAgent.n + 1))
        self.conformalSet = np.zeros_like(yhat)
        if self.hasf:
            sq = self.feng.getQuantileCombineOther(testX, self.fengList, self.fdreList, self.q, self.n, maxS=self.maxS, w=w)
        else:
            sq = self.feng.getQuantile(testX, self.q, self.n, maxS=self.maxS, w=w)
        sq_ = np.clip(sq, 0, self.maxS*2).reshape((-1,1))
        self.conformalSet[yhat < sq_] = 1.
        return self.conformalSet

if __name__ == '__main__':
    def generateXY(gamma, n):
        X = np.random.normal(0, 1, (n, 28, 28, 3))
        Y = np.random.randint(0, 7, (n, 1))
        return X, Y

    setseed(0)
    n = 100
    trainer = [MNISTTrainer(
        batch_size=64,
        learning_rate=0.001,
        num_epochs=10
    ) for i in range(2)]
    engTrAgent_ = [Agent(*generateXY(1, n))]+[Agent(*generateXY(1, 5*n))]
    engTrAgent = []
    for i in range(2):
        engTrAgent.append(Agent(trainer[i].covx(engTrAgent_[i].X), engTrAgent_[i].Y))
        engTrAgent[-1].loadS(calScoreCl(trainer[i], engTrAgent_[i].X, engTrAgent_[i].Y))
    calagent_ = Agent(*generateXY(1, n))
    calagent = Agent(trainer[0].covx(calagent_.X), calagent_.Y)
    calagent.loadS(calScoreCl(trainer[0], calagent_.X, calagent_.Y))

    hidden_dims_dre = [16]
    fcdre = FederatedCalibratedDRE(32, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=False)
    fcdre.trainOnXS(engTrAgent[0], engTrAgent[1:], caliN=5, epoches=50, log=log, mute=False)

    feng1 = FEng(d)
    feng1.train([engTrAgent[0]], epoches=300, bs=32, mute=False)

    feng2 = FEng(d)
    feng2.train(engTrAgent[1:], epoches=300, bs=32, mute=False)

    X_, Y = generateXY(1, 10*n)
    X = trainer[0].covx(X_)
    for k in [1,10]:
        setseed(0)
        fcp1 = FCPCl(trainer[0], calagent, feng1, [], [])
        fcp2 = FCPCl(trainer[0], calagent, feng1, [feng2]*k, [fcdre]*k)

        cs1 = fcp1.predict(X)
        cs2 = fcp2.predict(X)
        size1 = cs1.sum(-1).mean()
        size2 = cs2.sum(-1).mean()
        mar1 = cs1[np.arange(X.shape[0]), Y.squeeze()].mean()
        mar2 = cs2[np.arange(X.shape[0]), Y.squeeze()].mean()
        print(f"GLCP: mar={mar1:.4f}, size={size1:.4f}")
        print(f"FCP {k}: mar={mar2:.4f}, size={size2:.4f}")
