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
from CNNnet import *

class FedCPCl:
    def __init__(self, pred:MNISTTrainer, targetAgent:Agent, AgentList:list[Agent], alpha:float=.1):
        self.targetAgent = targetAgent
        self.AgentList = AgentList
        self.alpha = alpha
        self.predictor = pred
        self.digest = TDigest.from_values(self.targetAgent.getS().reshape(-1))
        self.calculateQuantile()

    def calculateQuantile(self):
        for agent in self.AgentList:
            self.digest += TDigest.from_values(agent.getS().reshape(-1))
        self.qhat = self.digest.quantile(1-self.alpha)

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

    setseed(0)
    fedcp = FedCPCl(trainer[0], calagent, engTrAgent, .1)
    cs = fedcp.predict(X)
    size = cs.sum(-1).mean()
    mar = cs[np.arange(X.shape[0]), Y.squeeze()].mean()
    print(f"{size:.4f}, {mar:.4f}")