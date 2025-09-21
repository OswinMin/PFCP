import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
import numpy as np
from Agent import *
from tools import *
from Predictor import *
from FDRE import *
from FEng import *
from FCP import *
from FedCP import *
from FedCPQQ import *
from color import *
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
import datetime
warnings.filterwarnings('ignore')
plt.rcParams['font.size'] = 18

def summation_real_one(Rpath, name, alpha=.1):
    cs, cov, size = loadData(Rpath, name)
    ind = np.load(f"{Rpath}/Test_Index.npy")
    mar = np.mean(cov)
    mean_size = np.mean(size)
    unique_ind = np.unique(ind)
    loc_cov = np.zeros(len(unique_ind))
    weight = np.zeros(len(unique_ind))
    for i in range(len(unique_ind)):
        loc_cov[i] = np.mean(cov[ind == unique_ind[i]])
        weight[i] = np.mean(ind == unique_ind[i])
    loc_err = (weight * np.abs(loc_cov - (1-alpha))).sum()
    return mar, mean_size, loc_err, loc_cov, weight

def summation_matrix(RpathList, nameList, alpha):
    ReList, LoccovList = [], []
    for Rpath, i in zip(RpathList, range(1,len(RpathList)+1)):
        l = summation_real_one(Rpath, nameList[0], alpha=alpha)[-1].shape[-1]
        Re = pd.DataFrame(np.zeros((len(nameList), 3)), columns=[f"mar", f"mis", f"size"], index=nameList)
        Loccov = pd.DataFrame(np.zeros((len(nameList)+1, l)), columns=range(1,l+1), index=nameList+['Weight'])
        for name, j in zip(nameList, range(len(nameList))):
            mar, mean_size, loc_err, loc_cov, weight = summation_real_one(Rpath, name, alpha=alpha)
            Re.iloc[j, :] = mar, loc_err, mean_size
            Loccov.iloc[j, :] = loc_cov - (1-alpha)
            if j == 0:
                Loccov.iloc[len(nameList), :] = weight
        ReList.append(Re)
        LoccovList.append(Loccov)
    return ReList, LoccovList

def man_con(df):
    str_con = pd.DataFrame(np.zeros_like(df, dtype=str), columns=[f"{x}" for x in df.columns], index=df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if i == np.argmin(df.iloc[:, j]):
                str_con.iloc[i, j] = f"$\\textbf{{{df.iloc[i, j]:.4f}}}$"
            else:
                str_con.iloc[i, j] = f"${df.iloc[i, j]:.4f}$"
    return str_con

def man_mar(df, nList):
    str_mar = pd.DataFrame(np.zeros_like(df, dtype=str), columns=[f"{x}" for x in df.columns], index=df.index)
    for i in range(df.shape[0]):
        for j in range(df.shape[1]):
            if abs(df.iloc[i,j]-(1-0.1)) >= 1/(1+nList[j]):
                str_mar.iloc[i,j] = f"$\\textbf{{{df.iloc[i,j]:.3f}}}\\times$"
            else:
                str_mar.iloc[i, j] = f"${df.iloc[i, j]:.3f}$"
    return str_mar

if __name__ == '__main__':
    RpathList = [f"../Result/Real_Protein/100_20_9_30_42_0.1", "../Result/Real_Bike/100_1_4_1_12_0.1", "../Result/Real_Crime/100_20_20_1_4_0.1", "../Result/Real_Achieve/100_8_20_3_2_20_1_0.1", "../Result/Real_Concrete/50_5_3_10_1_0.1","../Result/Real_Derma/200_500_4_0.03"]
    con_df = pd.DataFrame(np.zeros((6,6)), index=['GLCP', 'PFCP', "FedCP", "FedCP-QQ", "CPlab", "CPhet"], columns=["BIO", "BIKE", "CRIME", "STAR", "CONCRETE", "DERMA"])
    size_df = pd.DataFrame(np.zeros((6,6)), index=['GLCP', 'PFCP', "FedCP", "FedCP-QQ", "CPlab", "CPhet"], columns=["BIO", "BIKE", "CRIME", "STAR", "CONCRETE", "DERMA"])
    mar_df = pd.DataFrame(np.zeros((6,6)), index=['GLCP', 'PFCP', "FedCP", "FedCP-QQ", "CPlab", "CPhet"], columns=["BIO", "BIKE", "CRIME", "STAR", "CONCRETE", "DERMA"])
    nList = [100]*8+[50]*2+[200]*2
    alpha = 0.1
    nameList = ["LCP", "FCP", "FedCP", "FedCP-QQ", "CPlab", "CPhet"]
    ReList, LoccovList = summation_matrix(RpathList, nameList, alpha)
    for i in range(6):
        con_df.iloc[:, i] = ReList[i].iloc[:, 1]
        size_df.iloc[:, i] = ReList[i].iloc[:, 2]
        mar_df.iloc[:, i] = ReList[i].iloc[:, 0]

    str_con = man_con(con_df).T
    str_size = man_con(size_df).T
    str_mar = man_mar(mar_df, nList).T
    with open("real_sum.txt", 'w') as f:
        f.write(str_con.to_latex(column_format='c'*(len(str_con.columns)+1)))
        f.write("#"*50+"\n")
        f.write(str_mar.to_latex(column_format='c'*(len(str_con.columns)+1)))
        f.write("#"*50+"\n")
        f.write(str_size.to_latex(column_format='c'*(len(str_con.columns)+1)))











    # combinedResult = pd.concat(ReList, axis=1)
    # formatter = {col: '{:.3f}'.format for col in combinedResult.columns}
    # formatter_ = {col: '{:.2f}'.format for col in LoccovList[0].columns}
    # with open("result.txt", 'w') as f:
    #     f.write("Combined Result for alpha 0.1, 0.2, 0.3\n")
    #     f.write("#"*20+"\n")
    #     f.write(combinedResult.to_latex(float_format="$%.3f$"))
    #     # f.write(combinedResult.to_string(formatters=formatter))
    #     for loccov in LoccovList:
    #         f.write("#" * 20 + "\n")
    #         f.write(loccov.to_latex(float_format="$%.3f$"))
    #         # f.write(loccov.to_string(formatters=formatter_))