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
    df = pd.DataFrame(np.zeros((2,8)), columns=['mar','size','con','cta', 'mar_std','size_std','con_std','cta_std'], index=['GLCP','PFCP'])
    nL = ['LCP', 'FCP']
    for name, i in zip(nL,range(len(nL))):
        CS, COV, SIZE = loadData(Rpath, name)
        df.iloc[i, :] = summation(COV, SIZE, alpha)
    return df

if __name__ == '__main__':
    checkDir("../Figure")
    checkDir("../Figure/AuxN")
    markers = ['d', 'p']
    linestyle = ['dashed', 'solid']
    col = MyColor['dark'][:2]
    nameList = ['GLCP', 'PFCP']
    d, n, gammaSelectRule, g, seed = 20, 100, 0, .4, 1
    auxNList = [2,4,6,8,10,12,14,16,18,20]
    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    for s, k in zip(['abs', 'cos'], [1, 2]):
        RpathList = [f"../Result/AuxN_{s}/{d}_{n}_{auxN}_{gammaSelectRule}_{g}" for auxN in auxNList]
        ReList = [sum_path(rp, .1) for rp in RpathList]
        for name, j in zip(['GLCP','PFCP'], [0,1]):
            mar = [re.loc[name, 'mar'] for re in ReList]
            mar_std = [3*re.loc[name, 'mar_std'] for re in ReList]
            con = [re.loc[name, 'con'] for re in ReList]
            size = [re.loc[name, 'size'] for re in ReList]
            size_std = [3*re.loc[name, 'size_std'] for re in ReList]
            if name == 'GLCP':
                con = np.ones(len(con)) * np.mean(con)
                size = np.ones(len(size)) * np.mean(size)
            axes[k-1, 0].plot(auxNList, mar, color=col[j], linestyle=linestyle[j], marker=markers[j], linewidth=3, markersize=8, label=name)
            axes[k-1, 0].errorbar(auxNList, mar, yerr=mar_std, color=col[j], alpha=0.9, linestyle='none')
            axes[k - 1, 0].set_title(f"S{k}.5")
            axes[k - 1, 0].set_ylabel(f"Marginal Coverage")
            axes[k - 1, 0].set_ylim((0.8,1))
            axes[k-1, 1].plot(auxNList, con, color=col[j], linestyle=linestyle[j], marker=markers[j], linewidth=3, markersize=8, label=name)
            axes[k - 1, 1].set_title(f"S{k}.5")
            axes[k - 1, 1].set_ylabel(f"Miscoverage")
            axes[k-1, 2].plot(auxNList, size, color=col[j], linestyle=linestyle[j], marker=markers[j], linewidth=3, markersize=8, label=name)
            axes[k - 1, 2].errorbar(auxNList, size, yerr=size_std, color=col[j], alpha=0.9, linestyle='none')
            axes[k - 1, 2].set_title(f"S{k}.5")
            axes[k - 1, 2].set_ylabel(f"Size")
            print(f"{s}, {name}, {np.max(con)}, {np.max(size)}")
        axes[k - 1,0].set_xlabel("$K$")
        axes[k - 1,1].set_xlabel("$K$")
        axes[k - 1,2].set_xlabel("$K$")
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
    plt.subplots_adjust(top=0.88, bottom=0.1)
    plt.savefig(f"../Figure/AuxN/mar_cond.pdf", format='pdf')
    plt.show()