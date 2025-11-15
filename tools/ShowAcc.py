import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def Show():
    baselineFile ='/home/sunyi/MetaGF_TPAMI/Loss_landscapeAnalysis/tmpsave/onelayerbaseline/scores.tsv'
    baseline = np.loadtxt(baselineFile, delimiter='\t', skiprows=1)
    acclist=[]
    trainacclist=[]
    for i in range(0,baseline.shape[0]):
        acclist.append(baseline[i,5])
        trainacclist.append(baseline[i,4])

    fig, ax = plt.subplots()
    # ax.step(np.array(range(0,len(acclist))), np.stack(acclist), where='post', label='post')
    ax.plot(np.array(range(0,len(acclist))), np.stack(acclist), '--', color='black', alpha=0.3,label='nopcgrad')

    # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
    ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), '-', color='black', alpha=0.3,label='nopcgrad')

    baselineFile = '/home/sunyi/MetaGF_TPAMI/Loss_landscapeAnalysis/tmpsave/loss_landscape_onelayer2022-12-3-13-9-56/scores.tsv'
    baseline = np.loadtxt(baselineFile, delimiter='\t', skiprows=1)
    acclist = []
    trainacclist = []
    for i in range(0, baseline.shape[0]):
        acclist.append(baseline[i, 5])
        trainacclist.append(baseline[i, 4])

    ax.plot(np.array(range(0, len(acclist))), np.stack(acclist), '--', color='red', alpha=0.3)

    # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
    ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), '-', color='red', alpha=0.3)


    ax.grid(axis='x', color='0.95')
    ax.legend(title='Parameter where:')
    ax.set_frame_on(False)
    ax.set_title('matplotlib.axes.Axes.set_frame_on() Example')
    plt.savefig("comparison.png")
    plt.show()

def Show_List():
    namelist=['drmgf','ge','pcgrad','cagrad','sdn','nashmtl']
    Legennamelist = {'drmgf':"DR-MGF", 'ge':"GE", 'pcgrad':"Pcgrad", 'cagrad':"Cagrad", 'sdn':"baseline", 'nashmtl':"Nash-MTL"}
    color=['red','#FF9800','#757575','#FFEB3B','#448AFF','#009688', '#8BC34A','#DCEDC8']
    classnum = 100
    net = 'MSDnet'
    maxiter=300
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
        baselineFile =root+'/'+name+'/1/scores.tsv'
        if not os.path.exists(baselineFile):
            assert("no such file:{0}".format(baselineFile))
        baseline = np.loadtxt(baselineFile, delimiter='\t', skiprows=1)
        print(baselineFile)
        print(len(baseline))
        acclist=[]
        trainacclist=[]
        for i in range(0,maxiter):
            acclist.append(baseline[i,5])
            trainacclist.append(baseline[i,4])


        # ax.step(np.array(range(0,len(acclist))), np.stack(acclist), where='post', label='post')
        # ax.plot(np.array(range(0,len(acclist))), np.stack(acclist), '--', color=color[count], alpha=0.3,label=name)

        # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
        ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), color=color[count],label=Legennamelist[name])
        count+=1
    ax.grid(True, linestyle='--', alpha=0.4)
    # ax.grid(axis='x', linestyle='--', alpha=0.4)
    ax.legend()
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Classification Accuracy($\%$)')
    # ax.set_axis_on()
    # ax.set_frame_on(False)
    plt.savefig(root+"/comparison_convergence.png")
    plt.show()



def Show_List_multiexit():
    namelist=['DR-MGF','Meta-GF','DR-avgF','baseline']
    color = ['red', '#FF9800', '#757575', '#FFEB3B', '#448AFF', '#009688', '#8BC34A', '#DCEDC8']
    root='./'
    baselinelist=['../Baseline_res/20221129/results100/vgg/drmgf/1/scores.tsv',
                  '../Baseline_res/20221129/results100/vgg/meta/1/scores.tsv',
                  '../Baseline_res/20221129/results100/vgg16_sdn_meta/disentanglement-avgfusion/scores.tsv',
                  '../Baseline_res/20221129/results100/vgg/sdn/1/scores.tsv']
    plt.style.use(['../SciencePlots/scienceplots/styles/science.mplstyle',
                   '../SciencePlots/scienceplots/styles/journals/ieee.mplstyle'])
    fig, ax = plt.subplots()

    count=0
    for name in namelist:
        baselineFile =baselinelist[count]
        if not os.path.exists(baselineFile):
            assert("no such file:{0}".format(baselineFile))
        baseline = np.loadtxt(baselineFile, delimiter='\t', skiprows=1)
        print(baselineFile)
        print(len(baseline))
        acclist=[]
        trainacclist=[]
        for i in range(0,100):
            acclist.append(baseline[i,5])
            trainacclist.append(baseline[i,4])


        # ax.step(np.array(range(0,len(acclist))), np.stack(acclist), where='post', label='post')
        # ax.plot(np.array(range(0,len(acclist))), np.stack(acclist), '--', color=color[count], alpha=0.3,label=name)

        # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
        ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), color=color[count], alpha=0.7,label=name)
        count+=1

    ax.grid(True, linestyle='--', alpha=0.4)
    ax.legend()
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Classification Accuracy($\%$)')
    # ax.set_axis_on()
    # ax.set_frame_on(False)
    plt.savefig(root+"/comparison_convergence.png")
    plt.show()


def Show_List_multiexit_segnet():
    namelist=['DR-MGF','Meta-GF','DR-avgF','baseline']
    color=['red','#FF9800','#757575','#FFEB3B','#448AFF','#009688', '#8BC34A','#DCEDC8']
    root='./'
    baselinelist=['../E1_Multi_task/models/logs0305emaratio/cudacuda:2_SCALINGvalue_0.001_CONSTANT_SHARELAYER_0_losslandscape0_Metalr0.1_ema_0.9_aux_0.5_MetaGFstartEpoch100_UsingScale0_extraScaling_inverse0_weightdecay0.02023-3-5-4-19-17/valacclist.npy',
                  '../E1_Multi_task/models/logs0305emaratio/cudacuda:3_SCALINGvalue_0.001_CONSTANT_SHARELAYER_0_losslandscape0_Metalr0.1_ema_0.9_aux_0.5_MetaGFstartEpoch100_UsingScale0_extraScaling_inverse0_weightdecay0.02023-3-7-3-2-4/valacclist.npy',
                  '../E1_Multi_task/models/logs0305emaratio/cudacuda:4_SCALINGvalue_0.001_CONSTANT_SHARELAYER_0_losslandscape0_Metalr0.1_ema_0.0_aux_0.5_MetaGFstartEpoch-1_UsingScale0_extraScaling_inverse0_weightdecay0.02023-3-6-14-53-49/valacclist.npy',
                  '../E1_Multi_task/models/logs0305emaratio/cuda:cuda:0_baseline2023-3-8-2-53-13/valacclist.npy',]

    plt.style.use(['../SciencePlots/scienceplots/styles/science.mplstyle',
                   '../SciencePlots/scienceplots/styles/journals/ieee.mplstyle'])
    fig = plt.figure()
    ax1 = fig.add_axes([0.1, 0.7, 0.8, 0.25])
    ax2 = fig.add_axes([0.1, 0.4, 0.8, 0.25])
    ax3 = fig.add_axes([0.1, 0.1, 0.8, 0.25])
    count=0
    for name in namelist:
        baselineFile =baselinelist[count]
        if not os.path.exists(baselineFile):
            assert("no such file:{0}".format(baselineFile))
        baseline  = np.load(baselineFile)
        print(baselineFile)
        print(len(baseline))
        acclist=[]
        # left, bottom, width, height
        enpoch=200
        for task in range(0,3):
            trainacclist=baseline[task,:enpoch]
            if task==0:
                ax1.plot(np.array(range(0, trainacclist.shape[0])), trainacclist, color=color[count],label=name)
                ax1.set_ylabel('Segmentation Loss')
                ax1.set_axis_on()
                ax1.grid(axis='x', color='0.95')
            elif task==1:
                ax2.plot(np.array(range(0, trainacclist.shape[0])), trainacclist, color=color[count],label=name)
                ax2.set_ylabel('Depth Loss')
                ax2.set_axis_on()
                ax2.grid(axis='x', color='0.95')
            else:
                ax3.plot(np.array(range(0, trainacclist.shape[0])), trainacclist, color=color[count],label=name)
                ax3.set_ylabel('Normal Loss')
                ax3.set_axis_on()
                ax3.grid(axis='x', color='0.95')
            # ax.step(np.array(range(0,len(acclist))), np.stack(acclist), where='post', label='post')
            # ax.plot(np.array(range(0,len(acclist))), np.stack(acclist), '--', color=color[count], alpha=0.3,label=name)

            # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
            # ax.plot(np.array(range(0, trainacclist.shape[0])), trainacclist, '-', color=color[count], alpha=0.7,label=name)
        count+=1

    plt.grid(True, linestyle='--', alpha=0.4)
    plt.legend(fontsize=4,facecolor='blue')
    # plt.set_xlabel('Epochs', fontsize=15)
    # plt.set_ylabel('Classification Accuracy(%)', fontsize=15)
    # plt.set_axis_on()
    # plt.set_frame_on(False)
    plt.savefig(root+"/comparison_convergence_segnet.png")
    plt.show()

if __name__ == '__main__':
    # Show()
    # Show_List_multiexit()
    # Show_List_multiexit_segnet()
    Show_List()