import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def Show_List():
    namelist=['drmgf','ge','pcgrad','cagrad','sdn','nashmtl']
    # namelist = ['drmgf', 'ge', 'pcgrad', 'cagrad', 'nashmtl']
    Legennamelist = {'drmgf':"DR-MGF", 'ge':"GE", 'pcgrad':"Pcgrad", 'cagrad':"Cagrad", 'sdn':"baseline", 'nashmtl':"Nash-MTL"}

    color=['red','#FF9800','#757575','#FFEB3B','#448AFF','#009688', '#8BC34A','#DCEDC8']
    classnum = 100
    net = 'MSDnet'
    maxiter=100
    # range_limit=[50,200]
    range_limit = None
    if classnum == 10:
        root = "../Baseline_res/20221129/results/" + net  # "../Baseline_res/20221129/results/MSDnet"
        title = 'Budgeted batch classification on CIFAR-10'
    else:
        root = "../Baseline_res/20221129/results100/" + net  # "../Baseline_res/20221129/results/MSDnet"
        title = 'Budgeted batch classification on CIFAR-100'

    plt.style.use(['../SciencePlots/scienceplots/styles/science.mplstyle',
                   '../SciencePlots/scienceplots/styles/journals/ieee.mplstyle'])
    fig, ax = plt.subplots()

    count=0
    for name in namelist:
        baselineFile =root+'/'+name+'/1/latency.txt'
        if not os.path.exists(baselineFile):
            assert("no such file:{0}".format(baselineFile))
        baseline = np.loadtxt(baselineFile)
        baseline=baseline[::-1,:]
        # ax.step(np.array(range(0,len(acclist))), np.stack(acclist), where='post', label='post')
        # ax.plot(np.array(range(0,len(acclist))), np.stack(acclist), '--', color=color[count], alpha=0.3,label=name)

        # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
        ax.plot(baseline[:,0],baseline[:,1], color=color[count],label=Legennamelist[name],marker='+')
        count+=1
    ax.grid(True, linestyle='--', alpha=0.4)
    # ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend()
    ax.set_xlabel('Frame Per Second')
    ax.set_ylabel('Classification Accuracy($\%$)')
    # ax.set_axis_on()
    # ax.set_frame_on(False)

    plt.savefig(root + "/comparison_latency.png")
    plt.show()

if __name__ == '__main__':
    Show_List()