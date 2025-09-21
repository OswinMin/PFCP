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

class CPhet:
    def __init__(self, targetAgent:Agent, AgentList:list[Agent], fdre:FederatedCalibratedDRE, alpha:float=.1):
        self.targetAgent = targetAgent
        self.AgentList = AgentList
        self.fdre = fdre
        self.alpha = alpha
        self.predictor = self.targetAgent.pred
        self.calculateQuantile()

    def calculateQuantile(self):
        S = np.zeros(0)
        w = np.zeros(0)
        for agent in self.AgentList:
            S = np.concatenate((S, agent.getS().reshape(-1)))
            w = np.concatenate((w, self.fdre.ratio_XS(agent.X, agent.Y).reshape(-1)))
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
        yhat = self.predictor.predict(testX)
        self.conformalSet = np.ones((yhat.shape[0], 2))
        self.conformalSet[:, 0] = yhat.reshape(-1) - self.qhat
        self.conformalSet[:, 1] = yhat.reshape(-1) + self.qhat
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
    d = 10
    agentN = 5
    hidden_dims_pred = [30, 30]
    g = [1.]+[np.random.uniform(0.8,1) for i in range(agentN)]
    X, Y = generateRepXY(g[0], 200, d, 1000)
    agentL = [Agent(*generateXY(ga, n, d)) for ga in g]
    agentLT = [Agent(*generateXY(ga, n, d)) for ga in g]
    predL = []
    for ga, i in zip(g, range(len(g))):
        setseed(0)
        pred = Predictor(d, hidden_dims_pred)
        pred.trainFromAgent(agentLT[i])
        predL.append(pred)
        agentL[i].loadPred(pred)

    engTrAgent = [Agent(a.X, a.Y) for a in agentL]
    for a in engTrAgent:
        a.loadS(a.Y)
    fcdre = FederatedCalibratedDRE(d, [10], lr=5e-3, onXS=True, calibrate=False)
    fcdre.trainOnXS(engTrAgent[0], engTrAgent[1:], caliN=5, epoches=100, log=log, mute=False)

    setseed(0)
    cphet = CPhet(agentL[0], agentL, fcdre, .1)
    cs = cphet.predict(X)
    size = (cs[:, 1] - cs[:, 0]).mean()
    loc_cov = ((cs[:, [0]] <= Y) & (cs[:, [1]] >= Y)).mean(-1)
    mar = np.mean(loc_cov)
    print(f"{size:.4f}, {mar:.4f}")