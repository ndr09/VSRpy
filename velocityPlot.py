import numpy as np
import csv
from matplotlib import pyplot as plt
from plotFitness import *


def ordertest(A):
    return all(A[i] >= A[i + 1] for i in range(len(A) - 1))


def ordertest1(A):
    return all(A[i] <= A[i + 1] for i in range(len(A) - 1))


def ac(data):
    flag = [False for d in data]
    flag1 = [False for d in data]
    print("check monocity")
    for i in range(len(data)):
        print("check monocity " + str(i))
        for j in range(len(data[i])):
            if not (ordertest(data[i][j]) and ordertest1(data[i][j])):
                print(str(i) + "   " + str(j) + "  is not dec or increas")

    print(flag)
    print(flag1)

def miniBoxPlot(allff, shape):
    dataF = [allff[k]["flat"] for k in allff.keys()  if shape in str(k)]
    dataH = [allff[k]["hilly"] for k in allff.keys() if shape in str(k)]
    dataS = [allff[k]["steppy"] for k in allff.keys() if shape in str(k)]
    ls = [str(k).split("_")[2] for k in allff.keys() if shape in str(k)]
    fig, axs = plt.subplots(1,3)
    fig.tight_layout()
    fig.autofmt_xdate()
    fig.suptitle(shape)
    axs[0].boxplot(dataF, labels=ls)
    axs[0].set_xlabel("Flat")

    axs[1].boxplot(dataH, labels=ls)
    axs[1].set_xlabel("Hilly")

    axs[2].boxplot(dataS, labels=ls)
    axs[2].set_xlabel("Steppy")
    for a in axs:
        for tick in a.xaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')
    fig.subplots_adjust(top=0.85)
    plt.savefig(shape+"_val_test.png", dpi=800)
    plt.clf()


if __name__ == '__main__':
    shapes = ['biped']
    sensors = ['high']
    models = ['full']
    inits = ['zero']
    etas = ['001']
    baseDir = 'D:/dati hebbian'

    allData = dict()
    bests = dict()
    allVal = dict()
    k1 = list()
    k1f = list()
    for shape in shapes:
        for sensor in sensors:
            conf = shape + "_" + sensor + "_" + "MLP"
            allVal[conf] = dict()
            allVal[conf]["flat"] = list()
            allVal[conf]["hilly"] = list()
            allVal[conf]["steppy"] = list()
            for i in range(10):
                vel = pyworker.validatationSerializedA(String(getBest(
                    baseDir + "/" + shape + "/" +sensor+ "/"+"mlp/" + str(i) + ".txt")))
                allVal[conf]["flat"].append(vel[0])
                allVal[conf]["hilly"].append(np.median(vel[1:6]))
                allVal[conf]["steppy"].append(np.median(vel[6:]))
            for model in models:
                for eta in etas:
                    for init in inits:
                        conf = shape + "_" + sensor + "_" + model + "_" + eta + "_" + init
                        allVal[conf] = dict()
                        allVal[conf]["flat"] = list()
                        allVal[conf]["hilly"] = list()
                        allVal[conf]["steppy"] = list()
                        conf1 = shape + "_" + sensor + "_" + model+"-norm" + "_" + eta + "_" + init
                        allVal[conf1] = dict()
                        allVal[conf1]["flat"] = list()
                        allVal[conf1]["hilly"] = list()
                        allVal[conf1]["steppy"] = list()
                        for i in range(10):
                            vel = pyworker.validatationSerializedA(String(getBest(baseDir +"/"+shape + "/" + sensor + "/" + model + "/" + eta + "/" + init+"/" + str(i) + ".txt")))
                            allVal[conf]["flat"].append(vel[0])
                            allVal[conf]["hilly"].append(np.median(vel[1:6]))
                            allVal[conf]["steppy"].append(np.median(vel[6:]))

                            vel1 = pyworker.validatationSerializedA(String(getBest(
                                baseDir + "/rnorm/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/nb_" + str(
                                    i) + ".txt")))
                            allVal[conf1]["flat"].append(vel1[0])
                            allVal[conf1]["hilly"].append(np.median(vel1[1:6]))
                            allVal[conf1]["steppy"].append(np.median(vel1[6:]))


    for shape in shapes:
        miniBoxPlot(allVal, shape)

    #ac(de)
    #netPlot(sum(de) / len(de), "worm norm 0.01")
    '''
    for k in allData.keys():
        plt.plot(allData[k], label=str(k))
    plt.legend()
    cmap = plt.get_cmap("tab10")
    plt.axhline(10.0629, color=cmap(0))
    plt.axhline(5.983, color=cmap(1))
    plt.savefig("fit_biped.png", dpi=800)
    plt.clf()


    for shape in shapes:
        miniBoxPlot(allVal,shape)

    for shape in shapes:
        for sensor in sensors:
            conf = shape + "_" + sensor+"_mlp"
            for model in models:
                for eta in etas:
                    for init in inits:
                        conf = shape + "_" + sensor + "_" + model + "_" + eta + "_" + init
                        # netPlot(sum(bests[conf]) / len(bests[conf]), conf)
                        # netPlot(bests[conf][0], conf)'''
