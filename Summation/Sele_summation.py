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
    df = pd.DataFrame(np.zeros((6,8)), columns=['mar','size','con','cta', 'mar_std','size_std','con_std','cta_std'], index=['GLCP','PFCP','FedCP','FedCP-QQ', 'CPlab', 'CPhet'])
    nL = ['LCP', 'FCP', 'FedCP','FedCP-QQ', 'CPlab', 'CPhet']
    for name, i in zip(nL,range(len(nL))):
        CS, COV, SIZE = loadData(Rpath, name)
        df.iloc[i, :] = summation(COV, SIZE, alpha)
    return df

if __name__ == '__main__':
    checkDir("../Figure")
    checkDir("../Figure/Sele")
    n, agentN, d = 100, 20, 20
    markers = ['d', 'p', '*', 'o', 's', '^']
    linestyle = ['dashed', 'solid', 'dotted', 'dashdot', 'dotted', 'dashdot']
    col = MyColor['dark'][:6]
    corList = [0.1, 0.3, 0.5, 0.7, 0.9]
    seleList = [10, 20, 30]
    nameList = ['GLCP', 'PFCP', 'FedCP', 'FedCP-QQ', 'CPlab', 'CPhet']
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for sele in seleList:
        for s, k in zip(['abs', 'cos'], [1, 2]):
            rpList = [f"../Result/Sele_{s}/20_100_40_{sele}_{cor}" for cor in corList]
            reList = [sum_path(rp, .1) for rp in rpList]
            for name, j in zip(nameList, range(len(nameList))):
                mar = [re.loc[name, 'mar'] for re in reList]
                mar_std = [3*re.loc[name, 'mar_std'] for re in reList]
                axes[k - 1, sele // 10 - 1].plot(corList, mar, color=col[j], linestyle=linestyle[j], marker=markers[j], linewidth=4, markersize=10, label=name)
                axes[k - 1, sele // 10 - 1].errorbar(corList, mar, yerr=mar_std, color=col[j], alpha=0.9, linestyle='none')
            axes[k - 1, sele // 10 - 1].set_ylim((axes[k - 1, sele // 10 - 1].get_ylim()[0]-0.05, .93))
            axes[k - 1, sele // 10 - 1].set_title(f'S{k}.7 $n_{{\\rm sel}}={sele}$')
            axes[k - 1, sele // 10 - 1].set_xlabel('$p_{\\rm cor}$')
    axes[0,0].set_ylabel("Marginal Coverage")
    axes[1,0].set_ylabel("Marginal Coverage")
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
    plt.tight_layout(w_pad=0.3, h_pad=0.5)
    plt.subplots_adjust(top=0.86, bottom=0.12)
    plt.savefig(f"../Figure/Sele/marginal_p.pdf", format='pdf')
    plt.show()

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for sele in seleList:
        for s, k in zip(['abs', 'cos'], [1, 2]):
            rpList = [f"../Result/Sele_{s}/20_100_40_{sele}_{cor}" for cor in corList]
            reList = [sum_path(rp, .1) for rp in rpList]
            for name, j in zip(nameList[:2], range(len(nameList[:2]))):
                con = [re.loc[name, 'con'] for re in reList]
                axes[k - 1, sele // 10 - 1].plot(corList, con, color=col[j], linestyle=linestyle[j], marker=markers[j], linewidth=4, markersize=10, label=name)
            axes[k - 1, sele // 10 - 1].set_ylim((0.02, .09))
            axes[k - 1, sele // 10 - 1].set_title(f'S{k}.7 $n_{{\\rm sel}}={sele}$')
            axes[k - 1, sele // 10 - 1].set_xlabel('$p_{\\rm cor}$')
    axes[0, 0].set_ylabel("Miscoverage")
    axes[1, 0].set_ylabel("Miscoverage")
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
    plt.tight_layout(w_pad=0.3, h_pad=0.5)
    plt.subplots_adjust(top=0.86, bottom=0.12)
    plt.savefig(f"../Figure/Sele/miscoverage_p.pdf", format='pdf')
    plt.show()

    fig, axes = plt.subplots(2, 5, figsize=(20, 10))
    # for sele in seleList:
    for cor in corList:
        for s, k in zip(['abs', 'cos'], [1, 2]):
            rpList = [f"../Result/Sele_{s}/20_100_40_{sele}_{cor}" for sele in seleList]
            reList = [sum_path(rp, .1) for rp in rpList]
            for name, j in zip(['PFCP'], [1]):
                con = [re.loc[name, 'con'] for re in reList]
                axes[k - 1, int(5*cor-0.5)].plot(seleList, con, color=col[j], linestyle=linestyle[j], marker=markers[j], linewidth=4, markersize=10, label=name)
            # axes[k - 1, int(5*cor-0.5)].set_ylim((0.04, .06))
                m = (np.max(con)+np.min(con))/2
                axes[k - 1, int(5 * cor - 0.5)].set_ylim((m-0.004, m+0.004))
            axes[k - 1, int(5*cor-0.5)].set_title(f'S{k}.7 $p_{{\\rm cor}}={cor}$')
            axes[k - 1, int(5*cor-0.5)].set_xlabel('$n_{\\rm sel}$')
    axes[0, 0].set_ylabel("Miscoverage")
    axes[1, 0].set_ylabel("Miscoverage")
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
    plt.tight_layout(h_pad=0.5)
    plt.subplots_adjust(top=0.86, bottom=0.12)
    plt.savefig(f"../Figure/Sele/miscoverage_n.pdf", format='pdf')
    plt.show()