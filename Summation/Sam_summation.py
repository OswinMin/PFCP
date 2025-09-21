import sys
import os
sys.path.append(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')), 'Main'))
import pandas as pd
import numpy as np
from Agent import *
from tools import *
from color import *
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
    df = pd.DataFrame(np.zeros((4,8)), columns=['mar','size','con','cta', 'mar_std','size_std','con_std','cta_std'], index=['GLCP','PFCP','FedCP','FedCP-QQ'])
    nL = ['LCP', 'FCP', 'FedCP','FedCP-QQ']
    for name, i in zip(nL,range(len(nL))):
        CS, COV, SIZE = loadData(Rpath, name)
        df.iloc[i, :] = summation(COV, SIZE, alpha)
    return df

if __name__ == '__main__':
    checkDir("../Figure")
    checkDir("../Figure/Sam")
    n, agentN, d = 100, 20, 20
    markers = ['d', 'p', '*', 'o']
    linestyle = ['dashed', 'solid', 'dotted', 'dashdot']
    col = MyColor['dark'][:4]
    corList = [0.1, 0.3, 0.5, 0.7, 0.9]
    inds = np.arange(1, 41)
    nameList = ['GLCP','PFCP','FedCP','FedCP-QQ']
    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for s, k in zip(['abs', 'cos'], [1, 2]):
        for cor in corList:
            corN = int(cor*40)
            rp = f"../Result/Sam_{s}/20_100_40_{cor}/res_matrix_{cor}.npy"
            res = np.load(rp)
            res_mean = np.cumsum(res, axis=1) / np.array(range(1, res.shape[1]+1)).reshape((1, -1))
            av_res_mean = np.mean(res_mean, axis=0)
            av_res_std = 3 * np.std(res_mean, axis=0) / res_mean.shape[0] ** 0.5

            or_res = np.sort(res, axis=1)[:,::-1]
            or_mean = np.cumsum(or_res, axis=1) / np.array(range(1, or_res.shape[1]+1)).reshape((1, -1))
            av_or_res_mean = np.mean(or_mean, axis=0)
            av_or_res_std = 3 * np.std(or_mean, axis=0) / or_mean.shape[0] ** 0.5
            axes[k-1, int(cor*5-0.5)].grid()
            axes[k-1, int(cor*5-0.5)].plot(inds, av_or_res_mean, color=col[0], linestyle=linestyle[0], marker=markers[0], linewidth=3, markersize=3, label='Oracle Mean $\\gamma$')
            axes[k-1, int(cor*5-0.5)].errorbar(inds, av_or_res_mean, yerr=av_or_res_std, color=col[0], alpha=0.9, linestyle='none')
            axes[k-1, int(cor*5-0.5)].plot(inds, av_res_mean, color=col[1], linestyle=linestyle[1], marker=markers[1], linewidth=4, markersize=3, label='Selected Mean $\\gamma$')
            axes[k-1, int(cor*5-0.5)].errorbar(inds, av_res_mean, yerr=av_res_std, color=col[1], alpha=0.9, linestyle='none')
            axes[k - 1, int(cor * 5 - 0.5)].set_title(f"S{k}.7 $p_{{\\rm cor}}={cor}$")
            axes[k - 1, int(cor * 5 - 0.5)].set_ylim((0,1))
            axes[1, int(cor * 5 - 0.5)].set_xlabel("$K^\\prime$")
    axes[0,0].set_ylabel("mean $\\gamma$")
    axes[1,0].set_ylabel("mean $\\gamma$")
    handles1, labels1 = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles=handles1,
        labels=labels1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(labels1),
        frameon=False,
        fontsize=20
    )
    plt.tight_layout(w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(top=0.88, bottom=0.1)
    plt.savefig(f"../Figure/Sam/cum_mean.pdf", format='pdf')
    plt.show()


    fig, axes = plt.subplots(2, 5, figsize=(20, 9))
    for s, k in zip(['abs', 'cos'], [1, 2]):
        for cor in corList:
            corN = int(cor * 40)
            rp = f"../Result/Sam_{s}/20_100_40_{cor}/res_matrix_{cor}.npy"
            res = np.load(rp)
            av_res_mean = np.mean(res, axis=0)
            av_res_std = 3 * np.std(res, axis=0) / res.shape[0] ** 0.5

            or_res = np.sort(res, axis=1)[:, ::-1]
            av_or_res_mean = np.mean(or_res, axis=0)
            av_or_res_std = 3 * np.std(or_res, axis=0) / or_res.shape[0] ** 0.5
            axes[k - 1, int(cor * 5 - 0.5)].grid()
            axes[k - 1, int(cor * 5 - 0.5)].plot(inds, av_or_res_mean, color=col[0], linestyle=linestyle[0], marker=markers[0], linewidth=3, markersize=3, label='Oracle Mean $\\gamma$')
            axes[k - 1, int(cor * 5 - 0.5)].errorbar(inds, av_or_res_mean, yerr=av_or_res_std, color=col[0], alpha=0.9, linestyle='none')
            axes[k - 1, int(cor * 5 - 0.5)].plot(inds, av_res_mean, color=col[1], linestyle=linestyle[1], marker=markers[1], linewidth=4, markersize=3, label='Selected Mean $\\gamma$')
            axes[k - 1, int(cor * 5 - 0.5)].errorbar(inds, av_res_mean, yerr=av_res_std, color=col[1], alpha=0.9, linestyle='none')
            axes[k - 1, int(cor * 5 - 0.5)].set_title(f"S{k}.7 $p_{{\\rm cor}}={cor}$")
            axes[k - 1, int(cor * 5 - 0.5)].set_ylim((0, 1))
            axes[1, int(cor * 5 - 0.5)].set_xlabel("$K^\\prime$")
    axes[0, 0].set_ylabel("mean $\\gamma$")
    axes[1, 0].set_ylabel("mean $\\gamma$")
    handles1, labels1 = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles=handles1,
        labels=labels1,
        loc="upper center",
        bbox_to_anchor=(0.5, 1),
        ncol=len(labels1),
        frameon=False,
        fontsize=20
    )
    plt.tight_layout(w_pad=0.1, h_pad=0.1)
    plt.subplots_adjust(top=0.88, bottom=0.1)
    plt.savefig(f"../Figure/Sam/sin_mean.pdf", format='pdf')
    plt.show()