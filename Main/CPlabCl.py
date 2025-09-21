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
from CNNnet import *

class CPlabCl:
    def __init__(self, pred:MNISTTrainer, AgentList:list[Agent], fdre:FederatedCalibratedDRE, alpha:float=.1):
        self.AgentList = AgentList
        self.fdre = fdre
        self.alpha = alpha
        self.predictor = pred
        self.calculateQuantile()

    def calculateQuantile(self):
        S = np.zeros(0)
        w = np.zeros(0)
        for agent in self.AgentList:
            S = np.concatenate((S, agent.getS().reshape(-1)))
            w = np.concatenate((w, self.fdre.ratio_XS(np.zeros((agent.Y.shape[0],0)), agent.Y).reshape(-1)))
        self.qhat = self.weighted_quantile(S, w, self.alpha)

    def weighted_quantile(self, values, weights, alpha):
        sorted_indices = np.argsort(values)
        sorted_values = values[sorted_indices]
        sorted_weights = weights[sorted_indices]
        cum_weights = np.cumsum(sorted_weights)
        total_weight = cum_weights[-1]
        target_weight = (1-alpha) * total_weight
        idx = np.searchsorted(cum_weights, target_weight, side='right')
        if idx == 0:
            return sorted_values[0]
        elif idx >= len(sorted_values):
            return sorted_values[-1]
        return sorted_values[idx]

    def predict(self, testX):
        yhat = 1 - self.predictor.predict(testX)
        self.conformalSet = np.zeros_like(yhat)
        self.conformalSet[yhat < self.qhat] = 1.
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
    engTrAgent_ = [Agent(*generateXY(1, n))] + [Agent(*generateXY(1, 5 * n))]
    engTrAgent = []
    for i in range(2):
        engTrAgent.append(Agent(trainer[i].covx(engTrAgent_[i].X), engTrAgent_[i].Y))
        engTrAgent[-1].loadS(calScoreCl(trainer[i], engTrAgent_[i].X, engTrAgent_[i].Y))
    calagent_ = Agent(*generateXY(1, n))
    calagent = Agent(trainer[0].covx(calagent_.X), calagent_.Y)
    calagent.loadS(calScoreCl(trainer[0], calagent_.X, calagent_.Y))

    hidden_dims_dre = [16]
    X_, Y = generateXY(1, 10 * n)
    X = trainer[0].covx(X_)

    engTrAgent_ = [Agent(np.zeros((a.Y.shape[0],0)), a.Y) for a in engTrAgent]
    for a in engTrAgent_:
        a.loadS(a.Y)
    fcdre = FederatedCalibratedDRE(0, [10], lr=5e-3, onXS=True, calibrate=False)
    fcdre.trainOnXS(engTrAgent_[0], engTrAgent_[1:], caliN=5, epoches=100, log=log, mute=False)

    setseed(0)
    cplabcl = CPlabCl(trainer[0], engTrAgent[0:], fcdre, .1)
    cs = cplabcl.predict(X)
    size = cs.sum(-1).mean()
    mar = cs[np.arange(X.shape[0]), Y.squeeze()].mean()
    print(f"{size:.4f}, {mar:.4f}")