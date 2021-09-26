import numpy as np
from matplotlib import pyplot as plt
import csv
import os
from pprint import pprint as pp

os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))
os.environ[
    'CLASSPATH'] = "C:\\Users\\Mariella\\Desktop\\mapElites\\hebbianCode\\VSRpy\\test.jar"

import jnius_config

jnius_config.add_options('-Xrs', '-Xmx4096m')

from jnius import autoclass
import jnius

String = autoclass("java.lang.String")
terrain = "hilly-3-30-0"
shape = "biped-4x3"
sensors = "high_biped-0.01-f"
controller = "MLP-1-1-tanh"
duration = 60
Pyworker = autoclass('it.units.erallab.hmsrobots.Pyworker')

pyworker = Pyworker(String(terrain), String(shape), String(sensors),
                    String(controller), duration)


def loadData(filename):
    data = list()
    # print(filename)
    with open(filename, encoding='utf-8') as f:
        fr = csv.DictReader(f, delimiter=";")
        for l in fr:
            data.append(float(l["best→fitness→as[Outcome]→velocity"]))
    return data


def loadBreakable(filename, breakable):
    data = list()
    # print(filename)
    with open("./" + filename, encoding='utf-8') as f:
        fr = csv.DictReader(f, delimiter=";")
        for l in fr:
            if l['transformation'].startswith(breakable):
                data.append(float(l["outcome→velocity"]))
    return data


def loadVal(filename):
    data = dict()
    data["flat"] = list()
    data["hilly"] = list()
    data["steppy"] = list()
    # print(filename)
    with open(filename, encoding='utf-8') as f:
        fr = csv.DictReader(f, delimiter=";")
        for l in fr:
            if l["keys→validation.terrain"].startswith("flat"):
                data["flat"].append(float(l["outcome→velocity"]))
            if l["keys→validation.terrain"].startswith("hilly"):
                data["hilly"].append(float(l["outcome→velocity"]))
            if l["keys→validation.terrain"].startswith("steppy"):
                data["steppy"].append(float(l["outcome→velocity"]))
    data["flat"] = np.mean(data["flat"])
    data["hilly"] = np.mean(data["hilly"])
    data["steppy"] = np.mean(data["steppy"])
    return data


def getBest(filename):
    ser = ''
    v = 0
    with open(filename, encoding='utf-8') as f:
        fr = csv.DictReader(f, delimiter=";")
        for l in fr:
            ser = l["best→solution→serialized"]
            v = float(l["best→fitness→as[Outcome]→velocity"])
    print(v)
    return ser


def validateBreakable(bests, breakable):
    vals = pyworker.validatationSerializedBreakable(bests, String(breakable))
    return vals


def validate(dir, filename):
    vals = pyworker.validatationSerialized(String(getBest(dir + filename)))
    res = dict()
    res["flat"] = vals[0]
    res["hilly"] = np.mean(vals[1:6])
    res["steppy"] = np.mean(vals[6:])
    with open(dir + "val." + filename, "w", encoding='utf-8') as f:
        f.write("keys→validation.terrain;outcome→velocity\n")
        f.write("flat;" + str(vals[0]) + "\n")
        for i in range(1, 6):
            f.write("hilly;" + str(vals[i]) + "\n")
        for i in range(6, 11):
            f.write("steppy;" + str(vals[i]) + "\n")

    return res


def processData(data):
    median = list()
    tmp = list()
    for i in range(len(data)):
        tmp.append(len(data[i]))
    # print(tmp)
    for i in range(len(data[0])):
        tmp = list()
        for j in range(len(data)):
            tmp.append(data[j][i])
        median.append(np.median(tmp))
    return median


def valCheck(filename):
    return os.path.isfile(filename)

def boxPlotB(fig, y, shape, sensors, d, sensor):
    data = [
        d[shape+"_"+sensors[y]]["broken-0.1-0"],
        d[shape + "_" + sensors[y]+"_full"]["broken-0.1-0"],
        d[shape + "_" + sensors[y] + "_incoming"]["broken-0.1-0"],
        d[shape + "_" + sensors[y]]["broken-0.3-0"],
        d[shape + "_" + sensors[y] + "_full"]["broken-0.3-0"],
        d[shape + "_" + sensors[y] + "_incoming"]["broken-0.3-0"],
        d[shape + "_" + sensors[y]]["broken-0.5-0"],
        d[shape + "_" + sensors[y] + "_full"]["broken-0.5-0"],
        d[shape + "_" + sensors[y] + "_incoming"]["broken-0.5-0"]
    ]
    lab = []
    if y == 2:
        lab = ["DEW - 0.1", "F - 0.1", "I - 0.1", "DEW - 0.3", "F - 0.3", "I - 0.3","DEW - 0.5", "F - 0.5", "I - 0.5" ]
    else:
        lab = ["" for i in range(len(data))]
    box = fig[y].boxplot(data, labels=lab, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    fig[y].legend([box["boxes"][0], box["boxes"][1], box["boxes"][2]], ['DEW', 'Full', "Incoming"],
                  loc='upper right', fontsize="x-small")
    if y == 2:
        for tick in fig[y].xaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')
    fig[y].set_ylabel(sensor + '\n $\overline{v}$', multialignment='center')



def boxPlot(fig, d, conf, y, sensor):
    ''')

    colors = ['cyan', 'lightblue', 'lightgreen', 'tan', 'pink']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)

    '''
    data = [
        d[conf + "_mlp"]["flat"], d[conf + "_mlp"]["hilly"], d[conf + "_mlp"]["steppy"],
        d[conf + "_full_01_zero"]["flat"], d[conf + "_full_01_zero"]["hilly"], d[conf + "_full_01_zero"]["steppy"],
        d[conf + "_full_01_random"]["flat"], d[conf + "_full_01_random"]["hilly"],
        d[conf + "_full_01_random"]["steppy"],
        d[conf + "_full_001_zero"]["flat"], d[conf + "_full_001_zero"]["hilly"], d[conf + "_full_001_zero"]["steppy"],
        d[conf + "_full_001_random"]["flat"], d[conf + "_full_001_random"]["hilly"],
        d[conf + "_full_001_random"]["steppy"],
        d[conf + "_incoming_01_zero"]["flat"], d[conf + "_incoming_01_zero"]["hilly"],
        d[conf + "_incoming_01_zero"]["steppy"],
        d[conf + "_incoming_01_random"]["flat"], d[conf + "_incoming_01_random"]["hilly"],
        d[conf + "_incoming_01_random"]["steppy"],
        d[conf + "_incoming_001_zero"]["flat"], d[conf + "_incoming_001_zero"]["hilly"],
        d[conf + "_incoming_001_zero"]["steppy"],
        d[conf + "_incoming_001_random"]["flat"], d[conf + "_incoming_001_random"]["hilly"],
        d[conf + "_incoming_001_random"]["steppy"],
    ]
    lab = []
    if y == 2:
        lab = ["", "DEW", "", "", "F - 0.1 - Z", "", "", "F - 0.1 - R", "", "", "F - 0.01 - Z", "", "", "F - 0.01 - R",
               "",
               "", "I - 0.1 - Z", "", "", "I - 0.1 - R", "", "", "I - 0.01 - Z", "", "", "I - 0.01 - R", "", ]
    else:
        lab = ["" for i in range(len(data))]

    box = fig[y].boxplot(data, labels=lab, patch_artist=True)
    colors = ['lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan',
              'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan',
              'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan',
              'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan',
              'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan', 'lightblue', 'lightgreen', 'tan']
    for patch, color in zip(box['boxes'], colors):
        patch.set_facecolor(color)
    fig[y].legend([box["boxes"][0], box["boxes"][1], box["boxes"][2]], ['Flat', 'Hilly', "Steppy"],
                  loc='upper right', fontsize="x-small")
    if y == 2:
        for tick in fig[y].xaxis.get_major_ticks():
            tick.label.set_fontsize('xx-small')
    fig[y].set_ylabel(sensor + '\n $\overline{v}$', multialignment='center')


def netPlot(d, conf):
    fig, ax = plt.subplots()
    pos = ax.imshow(d, origin='lower')
    fig.colorbar(pos, ax=ax)
    fig.suptitle(conf)
    ax.set_xlabel("Tick")
    ax.set_ylabel("$\overline{v}$")
    plt.savefig("nets/norm/" + conf + ".png", dpi=1000)
    plt.clf()
    plt.close(fig)


def getSensors(serialized):
    observed = np.array(pyworker.locomoteSerialized(String(serialized)).getDataObservation())
    rf = len(observed) // len(observed[0])
    tmp = list()

    for i in range(len(observed)):
        tmp1 = list()
        for j in range(len(observed[i])):
            for k in range(rf):
                tmp1.append(observed[i][j])
        tmp.append(tmp1)

    observed = tmp
    npdata = np.array(observed)
    npdata = npdata.transpose()
    return npdata


def processNetData(data):
    out = list()
    for observed in data:
        rf = len(observed) // len(observed[0])
        tmp = list()

        for i in range(len(observed)):
            tmp1 = list()
            for j in range(len(observed[i])):
                for k in range(rf):
                    tmp1.append(observed[i][j])
            tmp.append(tmp1)

        observed = tmp
        npdata = np.array(observed)
        npdata = npdata.transpose()
        out.append(npdata)
    return out


def writeBest(best, filename):
    best.savefile(filename)


def findMin(data):
    mins = list()
    for k in data.keys():
        mins.append([np.min(data[k][i]) for i in range(10)])
    return np.min(mins)


def findMax(data):
    maxs = list()
    for k in data.keys():
        maxs.append([np.max(data[k][i]) for i in range(10)])
    return np.max(maxs)


def plotDataModel(fig, allData, conf, x, y, init, legend=False):
    # full
    # print(len(allData[conf + "_full_01_" + init]))
    if legend:
        fig[y, x].plot([i for i in range(len(allData[conf + "_full_01_" + init]))], allData[conf + "_full_01_" + init],
                       color='red', linestyle='dashed', label="Full - 0.1")
        fig[y, x].plot([i for i in range(len(allData[conf + "_full_001_" + init]))],
                       allData[conf + "_full_001_" + init],
                       color='blue', linestyle='dashed', label="Full - 0.01")
        # standard
        fig[y, x].plot([i for i in range(len(allData[conf]))], allData[conf],
                       color='black', label="DEW")
        # incoming
        fig[y, x].plot([i for i in range(len(allData[conf + "_incoming_01_" + init]))],
                       allData[conf + "_incoming_01_" + init],
                       color='red', linestyle='dotted', label="Incoming - 0.1")
        fig[y, x].plot([i for i in range(len(allData[conf + "_incoming_001_" + init]))],
                       allData[conf + "_incoming_001_" + init],
                       color='blue', linestyle='dotted', label="Incoming - 0.01")
    else:
        fig[y, x].plot([i for i in range(len(allData[conf + "_full_01_" + init]))], allData[conf + "_full_01_" + init],
                       color='red', linestyle='dashed')
        fig[y, x].plot([i for i in range(len(allData[conf + "_full_001_" + init]))],
                       allData[conf + "_full_001_" + init],
                       color='blue', linestyle='dashed')
        # standard
        fig[y, x].plot([i for i in range(len(allData[conf]))], allData[conf],
                       color='black')
        # incoming
        fig[y, x].plot([i for i in range(len(allData[conf + "_incoming_01_" + init]))],
                       allData[conf + "_incoming_01_" + init],
                       color='red', linestyle='dotted')
        fig[y, x].plot([i for i in range(len(allData[conf + "_incoming_001_" + init]))],
                       allData[conf + "_incoming_001_" + init],
                       color='blue', linestyle='dotted')


def plotDataInit(fig, allData, conf, x, y, legend=False):
    # full
    if legend:
        fig[y, x].plot([i for i in range(len(allData[conf + "_01_zero"]))], allData[conf + "_01_zero"],
                       color='red', linestyle='dashed', label="zero - 0.1")
        fig[y, x].plot([i for i in range(len(allData[conf + "_001_zero"]))], allData[conf + "_001_zero"],
                       color='blue', linestyle='dashed', label="zero - 0.01")
        # standard
        fig[y, x].plot([i for i in range(len(allData[conf[:-9]]))], allData[conf[:-9]],
                       color='black', label="DEW")
        # incoming
        fig[y, x].plot([i for i in range(len(allData[conf + "_01_random"]))],
                       allData[conf + "_01_random"],
                       color='red', linestyle='dotted', label="random - 0.1")
        fig[y, x].plot([i for i in range(len(allData[conf + "_001_random"]))],
                       allData[conf + "_001_random"],
                       color='blue', linestyle='dotted', label="random - 0.01")
    else:
        fig[y, x].plot([i for i in range(len(allData[conf + "_01_zero"]))], allData[conf + "_01_zero"],
                       color='red', linestyle='dashed')
        fig[y, x].plot([i for i in range(len(allData[conf + "_001_zero"]))], allData[conf + "_001_zero"],
                       color='blue', linestyle='dashed')
        # standard
        fig[y, x].plot([i for i in range(len(allData[conf[:-9]]))], allData[conf[:-9]],
                       color='black')
        # incoming
        fig[y, x].plot([i for i in range(len(allData[conf + "_01_random"]))],
                       allData[conf + "_01_random"],
                       color='red', linestyle='dotted')
        fig[y, x].plot([i for i in range(len(allData[conf + "_001_random"]))],
                       allData[conf + "_001_random"],
                       color='blue', linestyle='dotted')


if __name__ == '__main__':
    shapes = ['worm', 'biped']
    sensors = ['high']
    models = ['full']
    inits = ['zero']
    etas = ['001']
    baseDir = 'D:\dati hebbian/rnorm'
    allData = dict()
    allVal = dict()

    bests = dict()
    for shape in shapes:
        for sensor in sensors:
            if valCheck(baseDir + "/" + shape + "/" + sensor + "/mlp/val.0.txt"):
                val = dict()
                val["flat"] = list()
                val["hilly"] = list()
                val["steppy"] = list()
                for i in range(10):
                    tmp = loadVal(baseDir + "/" + shape + "/" + sensor + "/mlp/val." + str(i) + ".txt")
                    # print(tmp["flat"])
                    val["flat"].append(tmp["flat"])
                    val["hilly"].append(tmp["hilly"])
                    val["steppy"].append(tmp["steppy"])
                allVal[shape + "_" + sensor + "_mlp"] = val
            else:
                print(shape + "_" + sensor)
                val = dict()
                val["flat"] = list()
                val["hilly"] = list()
                val["steppy"] = list()
                for i in range(10):
                    res = validate(baseDir + "/" + shape + "/" + sensor + "/mlp/", str(i) + ".txt")
                    val["flat"].append(res["flat"])
                    val["hilly"].append(res["hilly"])
                    val["steppy"].append(res["steppy"])
                    allVal[shape + "_" + sensor + "_mlp"] = val

            data = list()
            best = list()
            for i in range(10):
                data.append(loadData(
                    baseDir + "/" + shape + "/" + sensor + "/mlp/" + str(i) + ".txt"))

            data = processData(data)
            allData[shape + "_" + sensor] = data

            if os.path.isfile(baseDir + "/" + shape + "/" + sensor + "/mlp/b." + str(i) + ".gz"):
                for i in range(10):
                    print("pre load " + baseDir + "/" + shape + "/" + sensor + "/mlp/" + str(i) + ".txt")
                    best.append(np.loadtxt(baseDir + "/" + shape + "/" + sensor + "/mlp/b." + str(i) + ".gz"))
                    print("post load " + baseDir + "/" + shape + "/" + sensor + "/mlp/" + str(i) + ".txt")
                bests[shape + "_" + sensor] = best
            else:
                for i in range(10):
                    best.append(String(
                        getBest(baseDir + "/" + shape + "/" + sensor + "/mlp/" + str(
                            i) + ".txt")))

                bests[shape + "_" + sensor] = processNetData(pyworker.locomoteSerializedParallel(best))
                for i in range(10):
                    np.savetxt(
                        baseDir + "/" + shape + "/" + sensor + "/mlp/b." + str(
                            i) + ".gz", bests[shape + "_" + sensor][i])

            for model in models:
                for eta in etas:
                    for init in inits:
                        if valCheck(
                                baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/val.0.txt"):
                            val = dict()
                            val["flat"] = list()
                            val["hilly"] = list()
                            val["steppy"] = list()
                            for i in range(10):
                                tmp = loadVal(
                                    baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/val." + str(
                                        i) + ".txt")
                                val["flat"].append(tmp["flat"])
                                val["hilly"].append(tmp["hilly"])
                                val["steppy"].append(tmp["steppy"])
                            allVal[shape + "_" + sensor + "_" + model + "_" + eta + "_" + init] = val
                        else:
                            print(shape + "_" + sensor + "_" + model + "_" + eta + "_" + init)
                            val = dict()
                            val["flat"] = list()
                            val["hilly"] = list()
                            val["steppy"] = list()
                            for i in range(10):
                                res = validate(
                                    baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/",
                                    str(
                                        i) + ".txt")
                                val["flat"].append(res["flat"])
                                val["hilly"].append(res["hilly"])
                                val["steppy"].append(res["steppy"])
                                allVal[shape + "_" + sensor + "_" + model + "_" + eta + "_" + init] = val

                        data = list()
                        best = list()
                        for i in range(10):
                            data.append(loadData(
                                baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/" + str(
                                    i) + ".txt"))
                        print(baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init)
                        newData = processData(data)
                        allData[shape + "_" + sensor + "_" + model + "_" + eta + "_" + init] = newData

                        if os.path.isfile(
                                baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/b." + str(
                                        i) + ".gz"):
                            for i in range(10):
                                print(
                                    "pre load " + baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/b." + str(
                                        i) + ".gz")
                                best.append(
                                    np.loadtxt(
                                        baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/b." + str(
                                            i) + ".gz"))
                                print(
                                    "post load " + baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/b." + str(
                                        i) + ".gz")
                            bests[shape + "_" + sensor + "_" + model + "_" + eta + "_" + init] = best
                        else:
                            for i in range(10):
                                best.append(String(getBest(
                                        baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/" + str(
                                            i) + ".txt")))
                            bests[shape + "_" + sensor + "_" + model + "_" + eta + "_" + init] = processNetData(
                                pyworker.locomoteSerializedParallel(best))
                            print(bests[shape + "_" + sensor + "_" + model + "_" + eta + "_" + init][i].shape)
                            for i in range(10):
                                np.savetxt(
                                    baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/b." + str(
                                        i) + ".gz",
                                    bests[shape + "_" + sensor + "_" + model + "_" + eta + "_" + init][i])

                        # plotData(shape+"_"+sensor+"_"+model+"_"+eta+"_"+init)

    for model in inits:
        fig, axs = plt.subplots(2, 3, sharey=True)
        for y in range(len(shapes)):
            for x in range(len(sensors)):
                if x == 1 and y == 1:
                    plotDataModel(axs, allData, shapes[y] + "_" + sensors[x], x, y, model, True)
                    # plotDataInit(axs, allData, shapes[y] + "_" + sensors[x] + "_incoming", x, y, True)
                else:
                    plotDataModel(axs, allData, shapes[y] + "_" + sensors[x], x, y, model, False)
                    # plotDataInit(axs, allData, shapes[y] + "_" + sensors[x] + "_incoming", x, y, False)
        fig.suptitle("Initilization " + model)
        axs[0, 0].set_ylabel('Worm \n $\overline{v}$', multialignment='center')
        axs[1, 0].set_ylabel('Biped\n $\overline{v}$', multialignment='center')
        axs[1, 0].set_xlabel('Low')
        axs[1, 1].set_xlabel('Medium')
        axs[1, 2].set_xlabel('High')
        fig.legend(loc='upper right', ncol=2, fontsize="x-small")
        plt.savefig("plots/"+"model_comparison_" + model + ".png", dpi=800)
        plt.clf()

    for model in models:
        fig, axs = plt.subplots(2, 3, sharey=True)
        for y in range(len(shapes)):
            for x in range(len(sensors)):
                if x == 1 and y == 1:
                    # plotDataModel(axs, allData, shapes[y] + "_" + sensors[x], x, y, model, True)
                    plotDataInit(axs, allData, shapes[y] + "_" + sensors[x] + "_incoming", x, y, True, shapes[y] + "_" + sensors[x])
                else:
                    # plotDataModel(axs, allData, shapes[y] + "_" + sensors[x], x, y, model, False)
                    plotDataInit(axs, allData, shapes[y] + "_" + sensors[x] + "_incoming", x, y, False, shapes[y] + "_" + sensors[x])
        fig.suptitle("model " + model)
        axs[0, 0].set_ylabel('Worm\n $\overline{v}$', multialignment='center')
        axs[1, 0].set_ylabel('Biped\n $\overline{v}$', multialignment='center')
        axs[1, 0].set_xlabel('Low')
        axs[1, 1].set_xlabel('Medium')
        axs[1, 2].set_xlabel('High')
        fig.legend(loc='upper right', ncol=2, fontsize="x-small")
        plt.savefig("plots/"+"init_comparison_" + model + ".png", dpi=800)
        plt.clf()

    for shape in shapes:
        fig, axs = plt.subplots(3, 1, sharey=True)
        fig.suptitle(shape)
        for y in range(len(sensors)):
            boxPlot(axs, allVal, shape + "_" + sensors[y], y, sensors[y][0].upper() + sensors[y][1:])
        plt.savefig("plots/"+shape + "_adaptation.png", dpi=800)
        plt.clf()

    # mi = findMin(bests)
    # ma = findMax(bests)
    # print(mi)
    # print(ma)

    for shape in shapes:
        for sensor in sensors:
            conf = shape + "_" + sensor
            netPlot(sum(bests[conf]) / len(bests[conf]), conf)
            for model in models:
                for eta in etas:
                    for init in inits:
                        conf = shape + "_" + sensor + "_" + model + "_" + eta + "_" + init
                        netPlot(sum(bests[conf]) / len(bests[conf]), conf)

    allValB = dict()
    allValBData = dict()
    breakables = ["broken-0.1-0", "broken-0.2-0", "broken-0.5-0"]

    for shape in shapes:
        for sensor in sensors:
            tmp = list()
            for i in range(10):
                tmp = getBest(baseDir + "/" + shape + "/" + sensor + "/mlp/" + str(i) + ".txt")
            allValB[shape + "_" + sensor] = dict()
            allValBData[shape + "_" + sensor] = dict()
            for breakable in breakables:
                allValB[shape + "_" + sensor][breakable] = validateBreakable(best, breakable)
                allValBData[shape + "_" + sensor][breakable] = processNetData(pyworker.locomoteSerializedParallelBreakable(best, breakable))
            for model in models:
                tmp = list()
                for i in range(10):
                    tmp = getBest(baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + "0.01" + "/" + "zero" + "/"+ str(i) + ".txt")
                allValB[shape + "_" + sensor+"_"+model] = dict()
                for breakable in breakables:
                    allValB[shape + "_" + sensor+"_"+model][breakable] = validateBreakable(best, breakable)
                    allValBData[shape + "_" + sensor+"_"+model][breakable] = processNetData(pyworker.locomoteSerializedParallelBreakable(best, breakable))

    for shape in shapes:
        for sensor in sensors:
            for breakable in breakables:
                conf = shape + "_" + sensor
                netPlot(sum(allValBData[conf]) / len(allValBData[conf]), conf+"_"+breakable.split("-")[1])
            for model in models:
                for breakable in breakables:
                    conf = shape + "_" + sensor + "_" + model
                    netPlot(sum(allValBData[conf]) / len(allValBData[conf]), conf+"_"+breakable.split("-")[1])

    for shape in shapes:
        fig, axs = plt.subplots(3, 1, sharey=True)
        fig.suptitle(shape)
        axs[0].set_title(shape)
        for y in range(len(sensors)):
            boxPlotB(axs,y, shape, sensors[y], allValB, sensors[y][0].upper() + sensors[y][1:])

        plt.savefig("plots/"+shape + "_breakable.png", dpi=800)
        plt.clf()