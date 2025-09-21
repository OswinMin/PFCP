from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from FDRE import *
from CNNnet import *
from tools import *

def combineAgents(AgentList:list[Agent]):
    if len(AgentList) == 1:
        return AgentList[0]
    d = AgentList[0].d
    X, Y, S = np.zeros((0, d)), np.zeros((0, 1)), np.zeros((0, 1))
    for agent in AgentList:
        X = np.concatenate((X, agent.getX()), axis=0)
        Y = np.concatenate((Y, agent.getY()), axis=0)
        try:
            S = np.concatenate((S, agent.getS()), axis=0)
        except:
            pass
    agent = Agent(X, Y)
    try:
        agent.loadS(S)
    except:
        pass
    return agent

def calScoreCl(pred:MNISTTrainer, X:np.ndarray, Y:np.ndarray):
    # X 是 28*28*3
    prob = pred.predict_prob(X)
    S = 1 - prob[np.arange(X.shape[0]), np.int_(Y.squeeze())]
    return S.reshape((-1,1))

def calScoreCl_(pred:MNISTTrainer, X:np.ndarray, Y:np.ndarray):
    # X 下一步由 classifier2 处理
    prob = pred.predict(X)
    S = 1 - prob[np.arange(X.shape[0]), Y.squeeze()]
    return S.reshape((-1,1))

class Agent:
    def __init__(self, X, Y, pred=None):
        self.X = X
        self.Y = Y.reshape((-1,1))
        self.n = X.shape[0]
        self.d = X.shape[1]
        self.pred = pred
        self.CDF = None
        self.S = None
        if self.pred is not None:
            self.calS()

    def loadCDF(self, cdf:CDF):
        """
        :param cdf: 要求 cdf 有 .conditional_cdf(x:np.ndarray, y:np.ndarray) 方法，计算 y | x 的条件分布，x : n * d   y : n * k ( k 一般是 1 )
        :return:
        """
        self.CDF = cdf

    def coverage(self, X:np.ndarray, CS:np.ndarray):
        if self.CDF is None:
            print(f"No oracle CDF, fails")
            return
        upper = self.CDF.conditional_cdf(X, CS[:, [1]])
        lower = self.CDF.conditional_cdf(X, CS[:, [0]])
        coverage = upper - lower
        return coverage

    def splitAgent(self, k):
        ind = np.random.permutation(self.n)
        X, Y = self.X[ind], self.Y[ind]
        agent1 = Agent(X[:k], Y[:k], pred=self.pred)
        agent2 = Agent(X[k:], Y[k:], pred=self.pred)
        if (self.pred is None) and (self.S is not None):
            S = self.S[ind]
            agent1.loadS(S[:k])
            agent2.loadS(S[k:])
        return agent1, agent2

    def loadPred(self, pred):
        self.pred = pred
        self.calS()

    def loadPredWithoutCalS(self, pred):
        self.pred = pred

    def loadS(self, S):
        self.S = S

    def getN(self):
        return self.n

    def getX(self):
        return self.X

    def getY(self):
        return self.Y

    def getX_T(self):
        return torch.tensor(self.X).float()

    def getY_T(self):
        return torch.tensor(self.Y).float()

    def calS(self):
        if self.pred is None:
            self.S = np.abs(self.Y)
        else:
            self.S = np.abs(self.Y-self.pred.predict(self.X))

    def getS(self):
        return self.S

    def getS_T(self):
        return torch.tensor(self.S).float()

    def getXS(self):
        return np.concatenate((self.X, self.S), axis=1)

    def getXSeq(self, ex:int=0):
        if ex == 0:
            return np.concatenate((self.X, self.S*np.ones_like(self.X)), axis=1)
        elif ex == 1:
            return np.concatenate((self.X, self.S * np.ones((X.shape[0], np.ceil(X.shape[1]/2)))), axis=1)
        else:
            return np.concatenate((self.X, self.S), axis=1)

    def getXS_T(self):
        return torch.tensor(np.concatenate((self.X, self.S), axis=1)).float()

    def scoreAgent(self, label:float=1.):
        XS = self.getXS()
        Y = np.ones((XS.shape[0],1)) * label
        return Agent(XS, Y)

    def XSAgent(self):
        return Agent(self.getX(), self.getS())

    def XYLoader(self, bs:int=32):
        dataset = torch.utils.data.TensorDataset(self.getX_T(), self.getY_T())
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
        return loader

    def XSLoader(self, bs:int=32):
        dataset = torch.utils.data.TensorDataset(self.getX_T(), self.getS_T())
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
        return loader

    def XLoader(self, label=1., bs:int=32):
        labels = torch.ones((self.X.shape[0], 1)) * label
        dataset = torch.utils.data.TensorDataset(self.getX_T(), labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
        return loader

    def combinedXSLoader(self, label=1., bs:int=32):
        XS = self.getXS_T()
        labels = torch.ones((XS.shape[0],1)) * label
        dataset = torch.utils.data.TensorDataset(XS, labels)
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
        return loader

    def getCenter(self):
        return self.X.mean(axis=0)

class FedAgent(Agent):
    def __init__(self, agent:Agent, pred=None, dre:FederatedCalibratedDRE=None):
        if (pred is None) and (agent.pred is None):
            super().__init__(agent.getX(), agent.getY())
        elif pred is not None:
            super().__init__(agent.getX(), agent.getY(), pred=pred)
        else:
            super().__init__(agent.getX(), agent.getY(), pred=agent.pred)
        self.calS()
        if dre is None:
            self.dre = unitWeight()
        else:
            self.dre = dre

    def ratioXS(self):
        self.R = self.dre.ratio(self.getXS()).reshape((-1,1))
        return self.R

    def ratioXS_T(self):
        return torch.tensor(self.ratioXS()).float()

    def XSrLoader(self, bs:int=32):
        dataset = torch.utils.data.TensorDataset(self.getX_T(), self.getS_T(), self.ratioXS_T())
        loader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
        return loader