from __future__ import annotations
from tools import *
# from engGenerator import *
from Agent import *
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

class Predictor(nn.Module):
    def __init__(self, inputDim:int, hiddenDim:list[int]):
        """
        :param inputDim: int
        :param hiddenDim: list
        """
        super(Predictor, self).__init__()
        self.inputDim = inputDim
        self.hiddenDim = hiddenDim
        self.layer_sizes = [inputDim] + hiddenDim + [1]
        self.layers = []
        for i in range(len(self.layer_sizes) - 1):
            self.layers.append(nn.Linear(self.layer_sizes[i], self.layer_sizes[i + 1]))
            if i < len(self.layer_sizes) - 2:
                self.layers.append(nn.LeakyReLU())
        self.model = nn.Sequential(*self.layers)

    def forward(self, x:torch.Tensor):
        """
        :param x: n*inputDim
        :return: n*1
        """
        return self.model(x)

    def predict(self, x:np.ndarray):
        """
        :param x: n*inputDim
        :return: n*1 np.ndarray
        """
        return self.forward(torch.tensor(x).float()).detach().numpy()

    def trainFromAgent(self, agent:Agent, batch_size:int=32, epochs:int=100, learning_rate:float=0.01, isLog=False, path='', log=None, mute=True):
        self.train(agent.getX(), agent.getY(), batch_size, epochs, learning_rate, isLog, path, log, mute)
        agent.loadPred(self)

    def train(self, X:np.ndarray, Y:np.ndarray, batch_size:int=32, epochs:int=100, learning_rate:float=0.01, isLog=False, path='', log=None, mute=True):
        """
        Use X, Y train a simple predictor
        :param X: n*inputDim
        :param Y: n*1
        :param batch_size:
        :param epochs:
        :param learning_rate:
        :return: No return
        """
        X_tensor = torch.tensor(X, dtype=torch.float32)
        Y_tensor = torch.tensor(Y, dtype=torch.float32)
        dataset = TensorDataset(X_tensor, Y_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            ttloss = 0
            t = 0
            for inputs, targets in dataloader:
                outputs = self(inputs)
                loss = criterion(outputs, targets)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                ttloss += loss.item()
                t += 1
            if not mute:
                if epoch % (epochs // 10) == max((epochs // 10) - 1, 1):
                    log(f'Training Predictor Epoch [{epoch + 1}/{epochs}], Loss: {ttloss / t:.4f}', path, isLog)

    def generateFeatureLoss(self, X:np.ndarray, Y:np.ndarray):
        Yhat = self.predict(X)
        Score = np.abs(Yhat - Y)
        return Score

    # def initiateEngressionRegressor(self, inputDim:int, hiddenDim:list[int], restrict:bool=False, randNum:int=1):
    #     self.engression = EngressionGenerator(inputDim, hiddenDim, restrict, randNum=randNum)
    #
    # def trainEngressionRegressor(self, X:np.ndarray, Y:np.ndarray, m:int=100, batch_size:int=32, epochs:int=100, learning_rate:float=0.01, isLog=False, path='', log=None, mute=True):
    #     Scores = self.generateFeatureLoss(X, Y)
    #     self.engression.train(X, Scores, m=m, batch_size=batch_size, epochs=epochs, learning_rate=learning_rate, isLog=isLog, path=path, log=log, mute=mute)

    def copy(self):
        predictor = Predictor(self.inputDim, self.hiddenDim)
        # try:
        #     predictor.engression = self.engression.copy()
        # except:
        #     pass
        predictor.load_state_dict(self.state_dict())
        return predictor


if __name__ == '__main__':
    n = 500
    d = 5
    X = np.random.normal(0, 1, (n, d))
    Y = X.sum(-1) + np.random.normal(0, 1, n) * np.cos(X.sum(-1))
    Y = Y.reshape((-1, 1))

    pred = Predictor(d, [10,10])
    pred.train(X, Y, mute=False, log=log)
    # pred.initiateEngressionRegressor(d, [20,20])
    # pred.trainEngressionRegressor(X, Y, m=50, mute=False, log=log)