from __future__ import annotations
import numpy as np
import torch
from numpy import floating
from typing import Union
from engression import engression
from Agent import *
from tools import *
from Predictor import *
from FDRE import *
from FCP import *

class FEng():
    def __init__(self, input_dim: int, num_layer=2, hidden_dim=100, noise_dim=100):
        super().__init__()
        self.input_dim = input_dim
        self.num_layer = num_layer
        self.hidden_dim = hidden_dim
        self.noise_dim = noise_dim

    def train(self, agentList:list[Agent], epoches:int=100, bs:int=32,  lr:float=1e-3, mute:bool=False):
        agent = combineAgents(agentList)
        X = agent.getX_T()
        S = agent.getS_T()
        if mute:
            with suppress_stdout():
                self.engressor = engression(X, S, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.noise_dim, lr=lr, num_epochs=epoches, batch_size=bs, device='cpu')
        else:
            self.engressor = engression(X, S, num_layer=self.num_layer, hidden_dim=self.hidden_dim, noise_dim=self.noise_dim, lr=lr, num_epochs=epoches, batch_size=bs, device='cpu')

    def generateN(self, x:np.ndarray, n:int=100) -> np.ndarray:
        if len(x.shape) == 1:
            x = x.reshape((1,-1))
        x = torch.tensor(x).float()
        y = self.engressor.sample(x, sample_size=n).view(x.shape[0], n).numpy()
        return y

    def percentile(self, x:np.ndarray, s:np.ndarray, n:int=100, maxS:float=-.0, w:np.ndarray=None):
        if w is None:
            w = np.zeros((x.shape[0], 1))
        if len(w.shape) == 1:
            w = w.reshape((-1,1))
        preds = self.generateN(x, n)
        weight = np.ones_like(preds)
        weight = weight / weight.shape[1] * (1 - w)
        preds = np.concatenate((preds, np.ones((preds.shape[0],1))*maxS), axis=1)
        weight = np.concatenate((weight, w), axis=1)
        perc = (weight * (preds <= s.reshape((-1,1)))).sum(-1)
        return perc

    def getQuantile(self, X:np.ndarray, q:Union[float,floating]=.9, n:int=100, maxS:float=-.0, w:np.ndarray=None):
        if w is None:
            w = np.zeros((X.shape[0], 1))
        if len(w.shape) == 1:
            w = w.reshape((-1,1))
        Yhats = self.generateN(X, n)
        weight = np.ones_like(Yhats)
        Yhats = np.concatenate((Yhats, np.ones((Yhats.shape[0], 1)) * maxS), axis=1)
        weight = weight / weight.shape[1] * (1 - w)
        weight = np.concatenate((weight, w), axis=1)
        qhat = empiricalQuantile(Yhats, weight, q)
        return qhat

    def calW(self, x:np.ndarray, X_C:np.ndarray, ave:float=.02):
        if len(X_C.shape) == 1:
            X_C = X_C.reshape((1,-1))
        if len(x.shape) == 1:
            x = x.reshape((1,-1))
        w = ((x-X_C)**2).sum(-1)
        std = w.std()
        w = np.exp(-w/std)
        w = np.clip(w / w.mean() * ave, 0, .1)
        return w

    def test(self, X:np.ndarray, Y:np.ndarray):
        Yhats = self.generateN(X, Y.shape[1])
        dis = [wasserstein_distance(Y[i,:], Yhats[i,:]) for i in range(Y.shape[0])]
        return dis

    def testQuantile(self, X:np.ndarray, Y:np.ndarray, q:float=.9, adjustq:float=.9):
        q = np.quantile(Y, q, axis=1)
        qhat = self.getQuantile(X, adjustq, Y.shape[1])
        return np.abs(q - qhat)

    def testCombineQuantile(self, X:np.ndarray, Y:np.ndarray, fengList:list[FEng], fdreList:list[FederatedCalibratedDRE], q:float=.9, adjustq:float=.9):
        q = np.quantile(Y, q, axis=1)
        qhat = self.getQuantileCombineOther(X, fengList, fdreList, adjustq, Y.shape[1])
        return np.abs(q - qhat)

    def generateNCombineOther(self, x:np.ndarray, n:int, fengList:list[FEng], fdreList:list[FederatedCalibratedDRE]):
        """
        :param x: N * d
        :param n:
        :param fengList: length [k]
        :param fdreList: length [k]
        :return: outs: N * nk, weights: N * nk
        """
        outs = self.generateN(x, n)
        weights = np.ones_like(outs)
        for feng, fdre in zip(fengList, fdreList):
            out = feng.generateN(x, n)
            weight = fdre.ratio_XS(x, out)
            outs = np.concatenate((outs, out), axis=1)
            weights = np.concatenate((weights, weight), axis=1)
        weights = weights / weights.sum(-1, keepdims=True)
        return outs, weights

    def percentileCombineOther(self, x:np.ndarray, s:np.ndarray, n:int, fengList:list[FEng], fdreList:list[FederatedCalibratedDRE], maxS:float=-.0, w:np.ndarray=None):
        """
        :param x: k*d
        :param s: k
        :param n: noise number
        :param fengList: federated engressor
        :param fdreList: federated dre
        :param maxS: max Score to tune local engressor
        :param w: weight for maxS   between [0,1]
        :return: 1-w weight for generated samples from engressor, w weight for upper bound maxS
        """
        if w is None:
            w = np.zeros((x.shape[0], 1))
        if len(w.shape) == 1:
            w = w.reshape((-1,1))
        outs, weights = self.generateNCombineOther(x, n, fengList, fdreList)
        weights = np.concatenate((weights*(1-w), w), axis=1)
        outs = np.concatenate((outs, np.ones((outs.shape[0],1))*maxS), axis=1)
        perc = (weights * (outs <= s.reshape((-1,1)))).sum(-1)
        return perc

    def getQuantileCombineOther(self, X:np.ndarray, fengList:list[FEng], fdreList:list[FederatedCalibratedDRE], q:Union[float,floating]=.9, n:int=100, maxS:float=-.0, w:np.ndarray=None):
        if w is None:
            w = np.zeros((X.shape[0], 1))
        if len(w.shape) == 1:
            w = w.reshape((-1,1))
        outs, weights = self.generateNCombineOther(X, n, fengList, fdreList)
        outs = np.concatenate((outs, np.ones((outs.shape[0], 1)) * maxS), axis=1)
        weights = np.concatenate((weights * (1 - w), w), axis=1)
        qhat = empiricalQuantile(outs, weights, q)
        return qhat

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
    d = 3
    epoches = 500

    gamma = np.random.uniform(0.8,1,10)
    predTrAgent = [Agent(*generateXY(1, n, d))] + [Agent(*generateXY(ga, n, d)) for ga in gamma]
    engTrAgent = [Agent(*generateXY(1, n, d))] + [Agent(*generateXY(ga, 2*n, d)) for ga in gamma]
    calagent = Agent(*generateXY(1, n, d))

    predictorList = [Predictor(d, hidden_dims_pred) for _ in predTrAgent]
    for i in range(len(predictorList)):
        predictorList[i].trainFromAgent(predTrAgent[i], epochs=200, mute=True)
        engTrAgent[i].loadPred(predictorList[i])
    calagent.loadPred(predictorList[0])

    X, Y = generateRepXY(1, 200, d, 1000)
    S = np.abs(Y-predictorList[0].predict(X))

    fcdre = FederatedCalibratedDRE(d, hidden_dims_dre, lr=5e-3, onXS=True, calibrate=False)
    fcdre.trainOnXS(engTrAgent[0], engTrAgent[1:], caliN=5, epoches=epoches, log=log, mute=False)

    feng1 = FEng(d)
    feng1.train([engTrAgent[0]], epoches=1000, bs=32, mute=False)
    adjustq1 = np.quantile(feng1.percentile(calagent.getX(), calagent.getS(), Y.shape[1]), .9)
    lossq1 = feng1.testQuantile(X, S, .9, adjustq=adjustq1).mean()

    feng2 = FEng(d)
    feng2.train(engTrAgent[1:], epoches=1000, bs=32, mute=False)
    adjustq2 = np.quantile(feng1.percentileCombineOther(calagent.getX(), calagent.getS(), Y.shape[1], [feng2], [fcdre]), .9)
    lossq2 = feng1.testCombineQuantile(X, S, [feng2], [fcdre], .9, adjustq=adjustq2).mean()

    print(lossq1, lossq2)