import numpy as np
from matplotlib import pyplot as plt
import csv
import os
from pprint import pprint as pp
import seaborn as sns
import pandas as pd

os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))
os.environ[
    'CLASSPATH'] = "C:\\Users\\Mariella\\Desktop\\mapElites\\hebbianCode\\VSRpy\\TwoDimHighlyModularSoftRobots.jar"

import jnius_config

jnius_config.add_options('-Xrs', '-Xmx4096m')

from jnius import autoclass
import jnius


def getBest(filename):
    ser = ''
    with open(filename, encoding='utf-8') as f:
        fr = csv.DictReader(f, delimiter=";")
        for l in fr:
            ser = l["best→solution→serialized"]
    return ser


def splitHB(hb):
    A = list()
    B = list()
    C = list()
    D = list()
    for i in range(0, len(hb), 4):
        A.append(hb[i])
        B.append(hb[i + 1])
        C.append(hb[i + 2])
        D.append(hb[i + 3])
    return A, B, C, D


def splitHBR(hb):
    A = list()
    B = list()
    C = list()
    D = list()
    for i in range(0, len(hb), 4):
        tmp = [np.abs(hb[i]), np.abs(hb[i + 1]), np.abs(hb[i + 2]), np.abs(hb[i + 3])]
        st = sum(tmp)
        tmp = [a / st for a in tmp]
        A.append(tmp[0])
        B.append(tmp[1])
        C.append(tmp[2])
        D.append(tmp[3])

    return A, B, C, D


def splitM(hb):
    mm = list()
    for i in range(0, len(hb), 4):
        tmp = [np.abs(hb[i]), np.abs(hb[i + 1]), np.abs(hb[i + 2]), np.abs(hb[i + 3])]
        st = sum(tmp)
        tmp = [a / st for a in tmp]
        mm.append(np.max(tmp))
    return mm


def plotDistSns(a, b, c, d, filename):
    fig, axs = plt.subplots(2, 2, sharey=True)
    fig.suptitle(filename.replace("_", " "))
    axs[0, 0].set_title("A")
    axs[0, 1].set_title("B")
    axs[1, 0].set_title("C")
    axs[1, 1].set_title("D")

    sns.stripplot(data=a, ax=axs[0, 0], jitter=1, alpha=.5).set_ylim([-10, 10])
    sns.stripplot(data=b, ax=axs[0, 1], jitter=1, alpha=.5).set_ylim([-10, 10])
    sns.stripplot(data=c, ax=axs[1, 0], jitter=1, alpha=.5).set_ylim([-10, 10])
    sns.stripplot(data=d, ax=axs[1, 1], jitter=1, alpha=.5).set_ylim([-10, 10])

    # axs[0, 0].set_yticks([])
    # axs[0, 1].set_yticks([])
    # axs[1, 0].set_yticks([])
    # axs[1, 1].set_yticks([])
    plt.savefig("dist/norm/sns_nb_" + filename + ".png", dpi=800)
    plt.clf()


def plotMaxSns(a, filename):
    fig, axs = plt.subplots(1, 1, sharey=True)
    fig.suptitle(filename.replace("_", " "))
    axs.set_title("A")

    sns.stripplot(data=a, ax=axs, jitter=1, alpha=.5).set_ylim([0, 1])

    # axs[0, 0].set_yticks([])
    # axs[0, 1].set_yticks([])
    # axs[1, 0].set_yticks([])
    # axs[1, 1].set_yticks([])
    plt.savefig("dist/norm/sns_max_" + filename + ".png", dpi=800)
    plt.clf()


def plotShapeDistSns(data, shape):
    fig, axs = plt.subplots(4, 1)
    fig.suptitle(shape + " incoming")
    c = 0
    axs[0].set_ylabel("A")
    axs[1].set_ylabel("B")
    axs[2].set_ylabel("C")
    axs[3].set_ylabel("D")
    al = list()
    bl = list()
    cl = list()
    dl = list()
    ll = list()
    a = pd.DataFrame()
    b = pd.DataFrame()
    c = pd.DataFrame()
    d = pd.DataFrame()
    for k in data.keys():
        if str(k).startswith(shape):
            l = data[k]

            lbls = str(k).split("_")[1:]
            ks = lbls[0].upper()[0] + "-" + lbls[2].upper() + "-" + lbls[3].upper()[0]

            ll.append(ks)
            al.append(np.array(l[0]).flat)
            bl.append(np.array(l[1]).flat)
            cl.append(np.array(l[2]).flat)
            dl.append(np.array(l[3]).flat)
            # print(str(k))
            # print(str(a.size)+"  "+str((np.abs(np.array(l[0]).flat) == 0).sum()))
            # print(str(a.size)+"  "+str((np.abs(np.array(l[1]).flat) == 0).sum()))
            # print(str(a.size)+"  "+str((np.abs(np.array(l[2]).flat) == 0).sum()))
            # print(str(a.size)+"  "+str((np.abs(np.array(l[3]).flat) == 0).sum()))
            # print("----------------")

    x1 = sns.stripplot(data=al, ax=axs[0], jitter=1, alpha=.1)
    x2 = sns.stripplot(data=bl, ax=axs[1], jitter=1, alpha=.1)
    x3 = sns.stripplot(data=cl, ax=axs[2], jitter=1, alpha=.1)
    x4 = sns.stripplot(data=dl, ax=axs[3], jitter=1, alpha=.1)

    x1.set_ylim([-10, 10])
    x2.set_ylim([-10, 10])
    x3.set_ylim([-10, 10])
    x4.set_ylim([-10, 10])
    print(x4)
    x1.set_xticklabels([])
    x2.set_xticklabels([])
    x3.set_xticklabels([])
    x4.set_xticklabels(ll)
    for tick in x4.xaxis.get_major_ticks():
        tick.label.set_fontsize('xx-small')
    plt.savefig("dist/norm/sns_nb_" + shape + "_full_dist.png", dpi=800)


String = autoclass("java.lang.String")
terrain = "hilly-3-30-0"
shape = "biped-4x3"
sensors = "high_biped-0.01-f"
controller = "MLP-0.1-0-tanh"
duration = 60
Pyworker = autoclass('it.units.erallab.hmsrobots.Pyworker')

pyworker = Pyworker(String(terrain), String(shape), String(sensors),
                    String(controller), duration)

if __name__ == '__main__':
    shapes = ['worm', 'biped']
    sensors = ['high']
    models = ['full']
    inits = ['zero']
    etas = ['001']
    baseDir = 'D:\dati hebbian/'
    allData = dict()
    allVal = dict()

    bests = dict()
    for shape in shapes:
        for sensor in sensors:
            for model in models:
                for eta in etas:
                    for init in inits:
                        A = list()
                        B = list()
                        C = list()
                        D = list()
                        print(shape + "/" + sensor + "/" + model + "/" + eta + "/" + init)
                        mm = list()
                        for i in range(10):
                            hb = pyworker.getHebbCoeff(String(getBest(
                                baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/" + str(
                                    i) + ".txt")))
                            a, b, c, d = splitHB(hb)
                            mm.append(splitM(hb))
                            print(np.min(mm[-1]))
                            A.append(a)
                            B.append(b)
                            C.append(c)
                            D.append(d)
                        bests[shape + "_" + sensor + "_" + model + "_" + eta + "_" + init] = [A, B, C, D]
                        plotMaxSns(mm, shape + "_" + sensor + "_" + model + "_" + eta + "_" + init)
                        plotDistSns(A, B, C, D, shape + "_" + sensor + "_" + model + "_" + eta + "_" + init)
    for shape in shapes:
        plotShapeDistSns(bests, shape)
