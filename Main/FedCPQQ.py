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

class FedCPQQ:
    def __init__(self, targetAgent: Agent, AgentList: list[Agent], alpha: float = .1):
        self.targetAgent = targetAgent
        self.AgentList = AgentList
        self.alpha = alpha
        self.predictor = self.targetAgent.pred
        self.agentN = len(self.AgentList) + 1
        self.n = np.min([a.n for a in self.AgentList]+[targetAgent.n])
        self.qhat = self.getQuantile()

    def callk(self, repN:int=100):
        setseed(0)
        self.M = np.zeros((self.agentN, self.n))
        for rep in range(repN):
            syn_data = np.random.uniform(0, 1, (self.agentN, self.n))
            innersorted_syn_data = np.sort(syn_data, axis=1)
            sorted_data = np.sort(innersorted_syn_data, axis=0)
            self.M += sorted_data
        self.M = self.M / repN
        M = self.M - (1-self.alpha)
        M[M<0] = 1.
        ind = np.argmin(M)
        l, k = int(ind // self.n), int(ind % self.n)
        return l, k

    def getQuantile(self, repN:int=1000):
        l, k = self.callk(repN=repN)
        qhatList = []
        for a in [self.targetAgent]+self.AgentList:
            s = a.getS().reshape(-1)[:self.n]
            qhatList.append(np.sort(s)[k])
        return np.sort(qhatList)[l]

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

    fedcpqq = FedCPQQ(agentL[0], agentL[0:], .1)
    cs = fedcpqq.predict(X)
    size = (cs[:, 1] - cs[:, 0]).mean()
    loc_cov = ((cs[:, [0]] <= Y) & (cs[:, [1]] >= Y)).mean(-1)
    mar = np.mean(loc_cov)
    print(f"{size:.4f}, {mar:.4f}")