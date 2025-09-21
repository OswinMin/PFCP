import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Union
import scipy.stats as stats
from sklearn.preprocessing import StandardScaler
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
plt.rcParams['font.size'] = 20

def summation_real_one(Rpath, name, alpha=.1):
    cs, cov, size = loadData(Rpath, name)
    ind = np.load(f"{Rpath}/Test_Index.npy")
    mar = np.mean(cov)
    mar_std = np.std(cov.mean(-1)) / cov.shape[0] ** 0.5
    mean_size = np.mean(size)
    mean_size_std = np.std(size.mean(-1)) / size.shape[0] ** 0.5
    unique_ind = np.unique(ind)
    loc_cov = np.zeros(len(unique_ind))
    loc_cov_std = np.zeros(len(unique_ind))
    weight = np.zeros(len(unique_ind))
    for i in range(len(unique_ind)):
        loc_cov[i] = np.mean(cov[ind == unique_ind[i]])
        loc_cov_std[i] = np.std(cov[ind == unique_ind[i]]) / (cov[ind == unique_ind[i]]).shape[0] ** 0.5
        weight[i] = np.mean(ind == unique_ind[i])
    loc_err = (weight * np.abs(loc_cov - (1-alpha))).sum()
    return mar, mean_size, mar_std, mean_size_std, loc_err, loc_cov, loc_cov_std, weight

def summation_matrix(RpathList, nameList, alpha):
    ReList, LoccovList = [], []
    l = summation_real_one(RpathList[0], nameList[0], alpha=alpha)[-1].shape[-1]
    for Rpath, i in zip(RpathList, range(1,len(RpathList)+1)):
        Re = pd.DataFrame(np.zeros((len(nameList), 5)), columns=[f"mar", f"mis", f"size", "mar_std", "size_std"], index=nameList)
        Loccov = pd.DataFrame(np.zeros((len(nameList)+1, l)), columns=range(1,l+1), index=nameList+['Weight'])
        for name, j in zip(nameList, range(len(nameList))):
            mar, mean_size, mar_std, mean_size_std, loc_err, loc_cov, loc_cov_std, weight = summation_real_one(Rpath, name, alpha=alpha)
            Re.iloc[j, :] = mar, loc_err, mean_size, mar_std, mean_size_std
            Loccov.iloc[j, :] = loc_cov - (1-alpha)
            if j == 0:
                Loccov.iloc[len(nameList), :] = weight
        ReList.append(Re)
        LoccovList.append(Loccov)
    return ReList, LoccovList

def summation_loc(RpathList, nameList):
    LoccovList = []
    LoccovStdList = []
    l = summation_real_one(RpathList[0], nameList[0], alpha=alpha)[-1].shape[-1]
    for Rpath, i in zip(RpathList, range(1,len(RpathList)+1)):
        Loccov = pd.DataFrame(np.zeros((len(nameList)+1, l)), columns=range(1,l+1), index=nameList+['Weight'])
        Loccov_std = pd.DataFrame(np.zeros((len(nameList), l)), columns=range(1,l+1), index=nameList)
        for name, j in zip(nameList, range(len(nameList))):
            mar, mean_size, mar_std, mean_size_std, loc_err, loc_cov, loc_cov_std, weight = summation_real_one(Rpath, name, alpha=alpha)
            Loccov.iloc[j, :] = 1 - loc_cov
            Loccov_std.iloc[j, :] = loc_cov_std
            if j == 0:
                Loccov.iloc[len(nameList), :] = weight
        LoccovList.append(Loccov)
        LoccovStdList.append(Loccov_std)
    return LoccovList, LoccovStdList

def math_format(x):
    if isinstance(x, (int, float)):
        return f"${x:.3f}$"
    return str(x)

def draw(ReList:list[pd.DataFrame], title, auxInd, savepath):
    markers = ['d', 'p', '*', 'o', 's', '^']
    linestyle = ['dashed', 'solid', 'dotted', 'dashdot', 'dotted', 'dashdot']
    col = MyColor['dark'][:6]

    FCP_cond = [re.loc['FCP', 'mis'] for re in ReList]
    LCP_cond = ReList[0].loc['LCP', 'mis']
    auxInd = [f"{x}" for x in auxInd]
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    axes[0].bar(auxInd, FCP_cond, color=col[1], linestyle=linestyle[1], label='FCP')
    axes[0].set_ylim((min(np.min(FCP_cond), np.max(FCP_cond)/3*2), axes[0].get_ylim()[1]))
    axes[0].set_xlabel("Number of Agents")
    axes[0].set_xticks(auxInd)
    axes[0].set_ylabel("Miscoverage")
    if LCP_cond < axes[0].get_ylim()[1]:
        axes[0].axhline(LCP_cond, color=col[0], linestyle=linestyle[0], linewidth=4, label='GLCP')
    axes[0].text(
        x=0.5,
        y=0.95,
        s=f"Miscoverage of GLCP = {LCP_cond:.3f}",
        ha="center",
        va="top",
        transform=axes[0].transAxes,
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    FCP_size = [re.loc['FCP', 'size'] for re in ReList]
    FCP_size_std = [2*re.loc['FCP', 'size_std'] for re in ReList]
    LCP_size = ReList[0].loc['LCP', 'size']
    axes[1].bar(auxInd, FCP_size, color=col[1], linestyle=linestyle[1], label='PFCP')
    axes[1].errorbar(auxInd, FCP_size, yerr=FCP_size_std, color='black', alpha=0.9, linestyle='none')
    axes[1].set_ylim((min(np.min(FCP_size), np.max(FCP_size) / 7 * 6), axes[1].get_ylim()[1]))
    axes[1].set_xlabel("Number of Agents")
    axes[1].set_xticks(auxInd)
    axes[1].set_ylabel("Size")
    if LCP_size < axes[1].get_ylim()[1]:
        axes[1].axhline(LCP_size, color=col[0], linestyle=linestyle[0], linewidth=4, label='GLCP')
    axes[1].text(
        x=0.5,
        y=0.95,
        s=f"Mean Size of GLCP = {LCP_size:.3f}",
        ha="center",
        va="top",
        transform=axes[1].transAxes,  # 使用相对坐标
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
    )

    axes[2].plot(auxInd, [re.loc['LCP', 'mar'] for re in ReList], color=col[0], linestyle=linestyle[0], marker=markers[0], linewidth=4, markersize=12, label='GLCP')
    axes[2].errorbar(auxInd, [re.loc['LCP', 'mar'] for re in ReList], [2*re.loc['LCP', 'mar_std'] for re in ReList], color=col[0], alpha=0.9, linestyle='none')
    axes[2].plot(auxInd, [re.loc['FCP', 'mar'] for re in ReList], color=col[1], linestyle=linestyle[1], marker=markers[1], linewidth=4, markersize=12, label='PFCP')
    axes[2].errorbar(auxInd, [re.loc['FCP', 'mar'] for re in ReList], [2 * re.loc['FCP', 'mar_std'] for re in ReList], color=col[1], alpha=0.9, linestyle='none')
    axes[2].plot(auxInd, [re.loc['FedCP', 'mar'] for re in ReList], color=col[2], linestyle=linestyle[2], marker=markers[2], linewidth=4, markersize=12, label='FedCP')
    axes[2].errorbar(auxInd, [re.loc['FedCP', 'mar'] for re in ReList], [2 * re.loc['FedCP', 'mar_std'] for re in ReList], color=col[2], alpha=0.9, linestyle='none')
    axes[2].plot(auxInd, [re.loc['FedCP-QQ', 'mar'] for re in ReList], color=col[3], linestyle=linestyle[3], marker=markers[3], linewidth=4, markersize=12, label='FedCP-QQ')
    axes[2].plot(auxInd, [re.loc['CPlab', 'mar'] for re in ReList], color=col[4], linestyle=linestyle[4], marker=markers[4], linewidth=4, markersize=12, label='CPlab')
    axes[2].errorbar(auxInd, [re.loc['CPlab', 'mar'] for re in ReList], [2 * re.loc['CPlab', 'mar_std'] for re in ReList], color=col[4], alpha=0.9, linestyle='none')
    axes[2].plot(auxInd, [re.loc['CPhet', 'mar'] for re in ReList], color=col[5], linestyle=linestyle[5], marker=markers[5], linewidth=4, markersize=12, label='CPhet')
    axes[2].errorbar(auxInd, [re.loc['CPhet', 'mar'] for re in ReList], [2 * re.loc['CPhet', 'mar_std'] for re in ReList], color=col[5], alpha=0.9, linestyle='none')
    axes[2].errorbar(auxInd, [re.loc['FedCP-QQ', 'mar'] for re in ReList], [2 * re.loc['FedCP-QQ', 'mar_std'] for re in ReList], color=col[3], alpha=0.9, linestyle='none')
    axes[2].set_xlabel("Number of Agents")
    axes[2].set_xticks(auxInd)
    axes[2].set_ylabel("Marginal Coverage")

    if LCP_cond < axes[0].get_ylim()[1]:
        handles1, labels1 = axes[0].get_legend_handles_labels()
    else:
        handles1, labels1 = axes[1].get_legend_handles_labels()
    handles2, labels2 = axes[2].get_legend_handles_labels()
    all_handles = handles1 + handles2
    all_labels = labels1 + labels2
    fig.legend(
        handles=all_handles,
        labels=all_labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=len(all_handles),
        frameon=False,
        fontsize=20
    )
    plt.tight_layout()
    plt.suptitle(title)
    plt.subplots_adjust(top=0.8, bottom=0.15)
    plt.savefig(savepath, format='pdf')
    plt.show()

def draw_loc(LoccovList, LoccovStdList, alphaList, savepath):
    col = MyColor['dark'][:4]
    linestyle = ['dashed', 'solid', 'dotted', 'dashdot']
    barwidth = 0.35
    Ind = LoccovList[0].columns.tolist()
    x1 = np.array(Ind) - barwidth/2
    x2 = np.array(Ind) + barwidth/2
    fig, axes = plt.subplots(1, 1, figsize=(13, 6))
    for i in range(1):
        LCP_cond = LoccovList[i].loc['LCP', :]
        LCP_std = LoccovStdList[i].loc['LCP', :]
        FCP_cond = LoccovList[i].loc['FCP', :]
        FCP_std = LoccovStdList[i].loc['FCP', :]
        weight = LoccovList[i].loc['Weight', :]
        LCP_mis = (np.abs(LCP_cond - alphaList[i]) * weight).sum()
        FCP_mis = (np.abs(FCP_cond - alphaList[i]) * weight).sum()
        axes.bar(x1, LCP_cond.values, width=barwidth, color=col[0], label='GLCP')
        axes.errorbar(x1, LCP_cond.values, yerr=LCP_std, color='black', alpha=0.9, linestyle='none')
        axes.bar(x2, FCP_cond.values, width=barwidth, color=col[1], label='PFCP')
        axes.errorbar(x2, FCP_cond.values, yerr=FCP_std, color='black', alpha=0.9, linestyle='none')
        axes.set_xticks(Ind)
        axes.axhline(alphaList[i], color='black', linestyle=linestyle[0], linewidth=4, label='Required')
        axes.text(
            x=0.5,
            y=0.95,
            s=f"Miscoverage of GLCP = {LCP_mis:.3f}\nMiscoverage of PFCP = {FCP_mis:.3f}",
            ha="center",
            va="top",
            transform=axes.transAxes,  # 使用相对坐标
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8)
        )
        # axes.set_title(f"$\\alpha={alphaList[i]}$")
        axes.set_xlabel("Subset Group Index")
        axes.set_ylabel("$1-$coverage")
    handles1, labels1 = axes.get_legend_handles_labels()
    fig.legend(
        handles=handles1,
        labels=labels1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(labels1),
        frameon=False,
        fontsize=20
    )
    plt.tight_layout()
    plt.subplots_adjust(top=0.88, bottom=0.15)
    plt.savefig(savepath, format='pdf')
    plt.show()

if __name__ == '__main__':
    checkDir("../Figure")
    checkDir("../Figure/Protein")
    for alpha in [0.1]:
        n, tar_ind, seed, X_int = 100, 9, 42, 30
        auxInd = [5,10,15,20,25,30,35,40]
        RpathList = [f"../Result/Real_Protein/{n}_{agentN}_{tar_ind}_{X_int}_{seed}_{alpha}" for agentN in [5,10,15,20,25,30,35,40]]
        nameList = ["LCP", "FCP", "FedCP", "FedCP-QQ", "CPlab", "CPhet"]
        ReList, _ = summation_matrix(RpathList, nameList, alpha)
        title = f'BIO ($n={n},\\alpha={alpha}$)'
        savepath = f"../Figure/Protein/nagent_{n}_{alpha}.pdf"
        draw(ReList, title, auxInd, savepath)

    Rpath = "../Result/Real_Protein/100_20_9_30_42_"
    nameList = ["LCP", "FCP", "FedCP", "FedCP-QQ"]
    alphaList = [0.1]
    LoccovList, LoccovStdList = summation_loc([Rpath + f"{a}" for a in alphaList], nameList)
    savepath = f"../Figure/Protein/conditional.pdf"
    draw_loc(LoccovList, LoccovStdList, alphaList, savepath)
















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