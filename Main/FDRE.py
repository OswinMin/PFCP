from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.linear_model import LogisticRegression
from typing import Union
from Agent import *
from tools import *
from Predictor import *

class FederatedCalibratedDRE(nn.Module):
    def __init__(self, input_dim: int, hidden_dims: list[int], lr:float=1e-3, onXS:bool=True, calibrate:bool=True):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        layers = []
        if onXS:
            prev_dim = input_dim + 1    # X,S 是 input_dim+1
        else:
            prev_dim = input_dim        # X 是 input_dim
        for dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, dim))
            layers.append(nn.LeakyReLU())
            prev_dim = dim
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())  # 二分类输出

        self.model = nn.Sequential(*layers)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.calibrate = calibrate

    def predict(self, X:np.ndarray):
        return self.model.forward(torch.tensor(X).float()).detach().numpy()

    def train(self, loader1, loader2List, n, m, epoches:int=100, isLog=False, path='', log=None, mute=True):
        for epoch in range(epoches):
            self.optimizer.zero_grad()
            tLoss = []
            for loader, N in zip([loader1]+loader2List, [n]+[m]*len(loader2List)):
                ttloss_ = 0
                for xs, y in loader:
                    yhat = self.model.forward(xs)
                    loss = torch.sum((yhat - y) ** 2) / N
                    loss.backward()

                    ttloss_ += loss.item()
                tLoss.append(ttloss_)
            self.optimizer.step()

            if not mute:
                if epoch % (epoches // 10) == max((epoches // 10) - 1, 1):
                    log(f'Training Classifier Epoch [{epoch + 1}/{epoches}], Loss on agent1: {tLoss[0]:.4f}, Loss on agent2: {np.sum(tLoss[1:]):.4f}',
                        path, isLog)

    def trainOnXS(self, agent1:Agent, agent2List:list[Agent], caliN:Union[int,float]=0.5, epoches:int=100, bs:int=32, isLog=False, path='', log=None, mute=True):
        """
        模拟在agent间传输数据，
        :param agent1: p(x) 1 类
        :param agent2: q(x) 0 类
        :param epoches:
        :return:
        """
        if self.calibrate:
            agent2List_ = agent2List
            agent2_List, agent2List = [], []
            if type(caliN) is int:
                agent1_, agent1 = agent1.splitAgent(caliN)
                for agent2 in agent2List_:
                    agent2_, agent2 = agent2.splitAgent(caliN)
                    agent2_List.append(agent2_)
                    agent2List.append(agent2)
            else:
                agent1_, agent1 = agent1.splitAgent(int(caliN*agent1.getN()))
                for agent2 in agent2List_:
                    agent2_, agent2 = agent2.splitAgent(int(caliN*agent2.getN()))
                    agent2_List.append(agent2_)
                    agent2List.append(agent2)
            self.train(agent1.combinedXSLoader(1.,bs), [agent2.combinedXSLoader(0., bs) for agent2 in agent2List], agent1.getN(), np.sum([agent2.getN() for agent2 in agent2List]), epoches, isLog, path, log, mute)
            self.calibrate_XS(agent1_, agent2_List)
        else:
            self.train(agent1.combinedXSLoader(1., bs), [agent2.combinedXSLoader(0., bs) for agent2 in agent2List], agent1.getN(), np.sum([agent2.getN() for agent2 in agent2List]), epoches, isLog, path, log, mute)

    def trainOnX(self, agent1:Agent, agent2:Agent, epoches:int=100, bs:int=32, isLog=False, path='', log=None, mute=True):
        """
        模拟在agent间传输数据，
        :param agent1: p(x) 1 类
        :param agent2: q(x) 0 类
        :param epoches:
        :return:
        """
        self.train(agent1.XLoader(1.,bs), agent2.XLoader(0., bs), agent1.getN(), agent2.getN(), epoches, isLog, path, log, mute)
        self.calibrate_X(agent1, agent2)

    def calibrate_XS(self, agent1:Agent, agent2List:list[Agent]):
        self.calibrator = LogisticRegression(class_weight='balanced', solver='lbfgs')
        Yhat1, Y1 = self.predict(agent1.getXS()), np.ones((agent1.getN(),1))
        Yhat2, Y2 = np.zeros((0,1)), np.zeros((0,1))
        for agent2 in agent2List:
            Yhat2_, Y2_ = self.predict(agent2.getXS()), np.zeros((agent2.getN(),1))
            Yhat2 = np.concatenate((Yhat2, Yhat2_), axis=0)
            Y2 = np.concatenate((Y2, Y2_), axis=0)
        Yhat = np.concatenate([Yhat1, Yhat2], 0)
        Y = np.concatenate([Y1, Y2], 0).reshape(-1)
        self.calibrator.fit(Yhat, Y)

    def calibrate_X(self, agent1:Agent, agent2:Agent):
        self.calibrator = LogisticRegression(class_weight='balanced', solver='lbfgs')
        Yhat1, Y1 = self.predict(agent1.getX()), np.ones((agent1.getN(),1))
        Yhat2, Y2 = self.predict(agent2.getX()), np.zeros((agent2.getN(),1))
        Yhat = np.concatenate([Yhat1, Yhat2], 0)
        Y = np.concatenate([Y1, Y2], 0).reshape(-1)
        self.calibrator.fit(Yhat, Y)

    def predict_calibrate(self, XS):
        """
        :param XS:
        :return: n*2 0类概率和1类概率
        """
        Yhat = self.predict(XS)
        return self.calibrator.predict_proba(Yhat)

    def ratio(self, XS):
        if self.calibrate:
            prob = self.predict_calibrate(XS)
            return prob[:, 1] / prob[:, 0]
        else:
            prob = self.predict(XS).reshape(-1)
            return prob / (1-prob)

    def ratio_XS(self, X, S):
        """
        :param X: n * d
        :param S: n * k
        :return:
        """
        n, k = X.shape[0], S.shape[1]
        X = np.repeat(X, S.shape[1], axis=0)    # nk * d
        S = S.reshape((-1,1))
        XS = np.concatenate([X, S], axis=1) # nk * (d+1)
        ratio = self.ratio(XS).reshape((n,k))
        return ratio

if __name__ == '__main__':
    def generateXY(gamma, n, d):
        X = np.random.normal(0, 1, (n, d))
        Y = X.sum(-1).reshape((-1, 1)) + np.random.normal(0, 1, (n, 1)) * np.cos(X.sum(-1)).reshape((-1, 1)) * gamma
        return X, Y
    np.random.seed(0)
    torch.manual_seed(0)
    n = 100
    m = 100
    hidden_dims_pred = [20, 20]
    hidden_dims_dre = []
    d = 2
    epoches = 2000

    agent1 = Agent(*generateXY(1, n, d))
    agent2 = Agent(*generateXY(1.5, m, d))
    agent3 = Agent(*generateXY(0.8, m, d))
    pred1 = Predictor(d, hidden_dims_pred)
    pred1.trainFromAgent(agent1)
    pred2 = Predictor(d, hidden_dims_pred)
    pred2.trainFromAgent(agent2)
    pred3 = Predictor(d, hidden_dims_pred)
    pred3.trainFromAgent(agent3)
    agent1_ = Agent(*generateXY(1, n, d))
    agent1_.loadPred(pred1)
    agent2_ = Agent(*generateXY(1.5, m, d))
    agent2_.loadPred(pred2)
    agent3_ = Agent(*generateXY(0.8, m, d))
    agent3_.loadPred(pred3)

    agentT2 = Agent(*generateXY(1.5, 1000, d))
    agentT2.loadPred(pred2)
    agentT3 = Agent(*generateXY(0.8, 1000, d))
    agentT3.loadPred(pred3)
    agentT = combineAgents([agentT2, agentT3])

    fcdre = FederatedCalibratedDRE(d, hidden_dims_dre, lr=1e-3, onXS=True, calibrate=False)
    fcdre.trainOnXS(agent1_, [agent2_, agent3_], epoches=epoches, log=log, mute=False)
    print(fcdre.ratio(agentT.getXS()).mean(), fcdre.ratio(agentT.getXS()).std())