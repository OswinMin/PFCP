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
    checkDir("../Figure/Auxx")
    n, agentN, d = 100, 20, 20
    markers = ['d', 'p', '*', 'o', 's', '^']
    linestyle = ['dashed', 'solid', 'dotted', 'dashdot', 'dotted', 'dashdot']
    col = MyColor['dark'][:6]
    nameList = ['GLCP','PFCP','FedCP','FedCP-QQ', 'CPlab', 'CPhet']
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    for s, k in zip(['abs', 'cos'], [1, 2]):
        for gammaSelectRule in [0, 1]:
            gList = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,0.95] if gammaSelectRule == 0 else [0.2,0.4,0.6,0.8,1.,1.4,1.8,2., 2.5, 3., 3.5, 4., 4.5, 5.]
            RpathList = [f"../Result/Aux_{s}/{d}_{n}_{agentN}_{gammaSelectRule}_{g}" for g in gList]
            reList = [sum_path(rp, 0.1) for rp in RpathList]
            for name, j in zip(nameList, range(len(nameList))):
                mar = [re.loc[name, 'mar'] for re in reList]
                mar_std = [3*re.loc[name, 'mar_std'] for re in reList]
                gList_ = gList if gammaSelectRule == 0 else [2*g for g in gList]
                axes[0, gammaSelectRule+(k-1)*2].plot(gList_, mar, color=col[j], linestyle=linestyle[j], marker=markers[j], linewidth=3, markersize=8, label=name)
                axes[0, gammaSelectRule+(k-1)*2].errorbar(gList_, mar, yerr = mar_std, color=col[j], alpha=0.9, linestyle='none')
                axes[0, gammaSelectRule+(k-1)*2].set_ylim((min(0.75, axes[0, gammaSelectRule+(k-1)*2].get_ylim()[0]), 1.02))
                if gammaSelectRule == 0:
                    axes[0, gammaSelectRule + (k - 1) * 2].set_xticks([0.2, 0.4, 0.6, 0.8, 1.])
                else:
                    axes[0, gammaSelectRule + (k - 1) * 2].set_xticks([2,4,6,8,10])
                axes[0, gammaSelectRule+(k-1) * 2].set_title(f"S{k}.{gammaSelectRule+3}")

                if name in ['GLCP', 'PFCP']:
                    con = [re.loc[name, 'con'] for re in reList]
                    axes[1, gammaSelectRule+(k-1) * 2].plot(gList_, con, color=col[j], linestyle=linestyle[j], marker=markers[j], linewidth=3, markersize=8, label=name)
                    if gammaSelectRule == 0:
                        axes[1, gammaSelectRule+(k-1) * 2].set_xticks([0.2,0.4,0.6,0.8,1.])
                        axes[1, gammaSelectRule+(k-1) * 2].set_xlabel('$c_1$')
                        axes[0, gammaSelectRule+(k-1) * 2].set_xlabel('$c_1$')
                    else:
                        axes[1, gammaSelectRule + (k - 1) * 2].set_xticks([2,4,6,8,10])
                        axes[1, gammaSelectRule + (k - 1) * 2].set_xlabel('$c_2$')
                        axes[0, gammaSelectRule + (k - 1) * 2].set_xlabel('$c_2$')
                    axes[1, gammaSelectRule+(k-1) * 2].set_title(f"S{k}.{gammaSelectRule + 3}")
                    axes[1, gammaSelectRule + (k - 1) * 2].set_ylim((0,0.1))
    axes[0,0].set_ylabel('Marginal Coverage')
    axes[1,0].set_ylabel('Miscoverage')
    handles1, labels1 = axes[0,0].get_legend_handles_labels()
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
    plt.savefig(f"../Figure/Auxx/mar_cond.pdf", format='pdf')
    plt.show()