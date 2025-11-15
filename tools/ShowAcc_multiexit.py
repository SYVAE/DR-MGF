import numpy as np
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import os

def Show():
    EXITID=0
    baselineFile ='../Baseline_res/20221129/results100/MSDnet/drmgf/1/valacclist.npy'
    acclist = np.load(baselineFile)
    plt.figure(1)
    plt.clf()

    # ax.step(np.array(range(0, len(acclist))), np.stack(acclist), where='post', label='post')
    plt.plot(np.array(range(0, len(acclist[EXITID]))), np.stack(acclist[EXITID]), '--', color='black', label="exit" + str(EXITID))

    baselineFile = '/home/user/DRMGF_0312/MetaGF_TPAMI/results100/msdnet_sdn_meta/baselineLRN_weightdecay5e-05_Inverse_deepcopy_EMA0.9_Auxlr0.4_MetaGFstartEpoch50_UsingScale0_extraScaling_adaptingClassifer2023-3-13-11-49-10/valacclist.npy'
    acclist = np.load(baselineFile)

    # ax.step(np.array(range(0, len(acclist))), np.stack(acclist), where='post', label='post')
    plt.plot(np.array(range(0, len(acclist[EXITID]))), np.stack(acclist[EXITID]), '--', color='red',
             label="exit" + str(EXITID))
    print(np.stack(acclist[EXITID]))
    plt.savefig("comparison.png")
    plt.show()

def Show_avg_multiexit():
    EXITID=2
    baselineFile ='../Baseline_res/20221129/results100/MSDnet/drmgf/1/valacclist.npy'
    acclist = np.load(baselineFile)
    plt.figure(1)
    plt.clf()
    acclist=np.mean(acclist,axis=0)
    # ax.step(np.array(range(0, len(acclist))), np.stack(acclist), where='post', label='post')
    plt.plot(np.array(range(0, len(acclist))), acclist, '--', color='black', label="exit" + str(EXITID))

    baselineFile = '/home/user/DRMGF_0312/MetaGF_TPAMI/results100/msdnet_sdn_meta/baselineLRN_weightdecay5e-05_Inverse_deepcopy_EMA0.9_Auxlr0.4_MetaGFstartEpoch50_UsingScale0_extraScaling_adaptingClassifer2023-3-13-11-49-10/valacclist.npy'

    # EXITID = 2
    # baselineFile = '../Baseline_res/20221129/results100/vgg/drmgf/1/valacclist.npy'
    # acclist = np.load(baselineFile)
    # plt.figure(1)
    # plt.clf()
    # acclist = np.mean(acclist, axis=0)
    # # ax.step(np.array(range(0, len(acclist))), np.stack(acclist), where='post', label='post')
    # plt.plot(np.array(range(0, len(acclist))), acclist, '--', color='black', label="exit" + str(EXITID))
    #
    # baselineFile = '/home/user/DRMGF_0312/MetaGF_TPAMI/results100/vgg16_nashmtl/cagrad2023-3-14-6-36-50/valacclist.npy'

    acclist = np.load(baselineFile)
    acclist = np.mean(acclist, axis=0)
    # ax.step(np.array(range(0, len(acclist))), np.stack(acclist), where='post', label='post')
    plt.plot(np.array(range(0, len(acclist))), acclist, '--', color='red',
             label="exit" + str(EXITID))
    # print(np.stack(acclist[EXITID]))
    plt.savefig("comparison.png")
    plt.show()


def Show_List():
    namelist=['ge','cagrad','pcgrad','meta','sdn']
    color = ['cyan', 'green', 'blue', 'red', 'black', 'yellow']
    root='/home/sunyi/ECCV_version_MetaGF/Baseline_res/results100/vgg/'
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
        for i in range(0,100):
            acclist.append(baseline[i,5])
            trainacclist.append(baseline[i,4])


        # ax.step(np.array(range(0,len(acclist))), np.stack(acclist), where='post', label='post')
        # ax.plot(np.array(range(0,len(acclist))), np.stack(acclist), '--', color=color[count], alpha=0.3,label=name)

        # ax.step(np.array(range(0, len(trainacclist))), np.stack(trainacclist), where='post', label='post')
        ax.plot(np.array(range(0, len(trainacclist))), np.stack(trainacclist), '-', color=color[count], alpha=0.3,label=name)
        count+=1

    ax.grid(axis='x', color='0.95')
    ax.legend(title='Method:')
    ax.set_xlabel('Epochs', fontsize=15)
    ax.set_ylabel('Classification Accuracy(%)', fontsize=15)
    ax.set_axis_on()
    ax.set_frame_on(False)
    plt.savefig(root+"/comparison_convergence.png")
    plt.show()


if __name__ == '__main__':
    Show_avg_multiexit()
    # Show()