from __future__ import annotations
import numpy as np
from typing import Union
from Agent import *
from tools import *
from Predictor import *
from sklearn.cluster import KMeans

class FClus:
    def __init__(self, tar_agent:Agent, aux_agentList:list[Agent], n:int=10, ex:int=0):
        self.tar_agent = tar_agent
        self.aux_agentList = aux_agentList
        self.n = n
        self.ex = ex
        self.agentList = [self.tar_agent] + aux_agentList
        self.kmList, self.weight = [], []
        for agent in self.agentList:
            km, w = self.local_cluster(agent, self.n)
            self.kmList.append(km)
            self.weight.append(w)
        self.global_km, self.global_clus, self.global_weight = self.global_cluster(self.kmList, self.weight, self.n)
        self.agent_feature = [self.local_feature(agent, self.global_km) for agent in self.agentList]
        self.aux_distance = [np.abs(self.agent_feature[0] - fea).sum() for fea in self.agent_feature[1:]]
        self.sortAux()

    def sortAux(self):
        indice = np.argsort(self.aux_distance)
        self.aux_agentList = [self.aux_agentList[i] for i in indice]
        self.aux_distance = np.array(self.aux_distance)[indice]

    def local_cluster(self, agent:Agent, n:int=10):
        XS = agent.getXSeq(self.ex)
        self.d = XS.shape[1]
        km = KMeans(n_clusters=n)
        km.fit(XS)
        return km, agent.n

    def local_feature(self, agent:Agent, km:KMeans):
        distance = km.transform(agent.getXSeq(self.ex))
        distance = distance - distance.mean(-1, keepdims=True)
        logit = np.exp(-distance)
        logit = logit / logit.sum(-1, keepdims=True)
        feature = logit.sum(0)
        return feature / feature.sum()

    def global_cluster(self, kmList, weight, n:int=10):
        clus = np.zeros((0, self.d))
        weights = np.zeros(0)
        for km, w in zip(kmList, weight):
            clus = np.concatenate((clus, km.cluster_centers_), axis=0)
            weights = np.concatenate((weights, w*np.ones(km.cluster_centers_.shape[0])), axis=0)
        weights = weights / np.sum(weights)
        km = KMeans(n_clusters=n)
        km.fit(clus, sample_weight=weights)
        return km, clus, weights

    def getKImportant(self, k):
        return self.aux_agentList[:k]

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
    epoches = 300

    gamma = np.random.uniform(0.2,1,20)
    predTrAgent = [Agent(*generateXY(1, n, d))] + [Agent(*generateXY(ga, n, d)) for ga in gamma]
    engTrAgent = [Agent(*generateXY(1, n, d))] + [Agent(*generateXY(ga, 2*n, d)) for ga in gamma]
    calagent = Agent(*generateXY(1, n, d))
    for i in range(len(engTrAgent)-1):
        engTrAgent[i+1].gamma = gamma[i]
        predTrAgent[i+1].gamma = gamma[i]

    predictorList = [Predictor(d, hidden_dims_pred) for _ in predTrAgent]
    for i in range(len(predictorList)):
        predictorList[i].trainFromAgent(predTrAgent[i], epochs=200, mute=True)
        engTrAgent[i].loadPred(predictorList[i])
    calagent.loadPred(predictorList[0])

    fclus = FClus(engTrAgent[0], engTrAgent[1:], 10, 0)
    # np.corrcoef([a.gamma for a in fclus.aux_agentList], range(len(fclus.aux_agentList)))
    for a, i in zip(fclus.aux_agentList, range(len(fclus.aux_agentList))):
        print(f"{i}: {a.gamma:.4f}")