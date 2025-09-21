import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
import pandas as pd
import numpy as np
from Agent import *
from tools import *
from color import *
from itertools import product
import matplotlib.pyplot as plt
import warnings
import datetime
warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 20

def summation(COV, SIZE, alpha:float=.1):
    """
    :param COV: repeats, testN
    :param SIZE: repeats, testN
    """
    mar = np.mean(COV)
    mar_ = np.std(COV.mean(-1)) / COV.shape[0] ** 0.5
    size = np.mean(SIZE)
    size_ = np.std(SIZE.mean(-1)) / COV.shape[0] ** 0.5
    local_mar = COV.mean(0)
    local_cov = np.mean(np.abs(local_mar-(1-alpha)))
    local_cov_ = np.std(np.abs(local_mar-(1-alpha))) / COV.shape[0] ** 0.5
    tt_local_cov = np.mean(np.abs(COV-(1-alpha)))
    tt_local_cov_ = np.std(np.abs(COV-(1-alpha))) / COV.shape[0] ** 0.5
    return mar, size, local_cov, tt_local_cov, mar_, size_, local_cov_, tt_local_cov_


def sum_path(Rpath, alpha):
    df = pd.DataFrame(np.zeros((5,8)), columns=['mar','size','con','cta', 'mar_std','size_std','con_std','cta_std'], index=['GLCP','PFCP1','PFCP2','FedCP','FedCP-QQ'])
    nL = ['LCP', 'FCP', 'FCP_','FedCP','FedCP-QQ']
    for name, i in zip(nL,range(len(nL))):
        CS, COV, SIZE = loadData(Rpath, name)
        df.iloc[i, :] = summation(COV, SIZE, alpha)
    return df

if __name__ == '__main__':
    checkDir("../Figure")
    checkDir("../Figure/CD")
    n, agentN, g = 100, 20, 0.2
    markers = ['d', 'p', '^', '*', 'o']
    linestyle = ['dashed', 'solid', 'solid', 'dotted', 'dashdot']
    col = MyColor['dark'][:2]+[MyColor['dark'][6]]+MyColor['dark'][2:4]
    nameList = ['GLCP','PFCP1','PFCP2']
    dList = [10,15,20,25,30]
    sizetable = pd.DataFrame(index=nameList*2, columns=dList*2)
    for s in ['abs', 'cos']:
        si = 0 if s=='abs' else 1
        for gammaSelectRule in [0, 1]:
            RpathList = [f"../Result/CD_{s}/{d}_{n}_{agentN}_{gammaSelectRule}_{g}" for d in dList]
            reList = [sum_path(rp, 0.1) for rp in RpathList]
            for namei in range(len(nameList)):
                for di in range(len(dList)):
                    sizetable.iloc[si*len(nameList)+namei, gammaSelectRule*len(dList)+di] = reList[di].iloc[namei, 1]

    for i in range(2):
        for j in range(3):
            s = f"{nameList[j]} & " + " & ".join([f"$\\textbf{{{x:.2f}}}$" for x in sizetable.iloc[3*i+j,:]]) + r"\\"
            print(s)
            if j > 0:
                s = " & " + " & ".join([f"${100*(1-x):.2f}\\%$" for x in sizetable.iloc[3*i+j,:]/sizetable.iloc[3*i,:]]) + r"\\"
                print(s)