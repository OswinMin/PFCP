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
from fastdigest import TDigest

class FedCP:
    def __init__(self, targetAgent:Agent, AgentList:list[Agent], alpha:float=.1):
        self.targetAgent = targetAgent
        self.AgentList = AgentList
        self.alpha = alpha
        self.predictor = self.targetAgent.pred
        self.digest = TDigest.from_values(self.targetAgent.getS().reshape(-1))
        self.calculateQuantile()

    def calculateQuantile(self):
        for agent in self.AgentList:
            self.digest += TDigest.from_values(agent.getS().reshape(-1))
        self.qhat = self.digest.quantile(1-self.alpha)

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

    setseed(0)
    fedcp = FedCP(agentL[0], agentL[0:], .1)
    cs = fedcp.predict(X)
    size = (cs[:, 1] - cs[:, 0]).mean()
    loc_cov = ((cs[:, [0]] <= Y) & (cs[:, [1]] >= Y)).mean(-1)
    mar = np.mean(loc_cov)
    print(f"{size:.4f}, {mar:.4f}")