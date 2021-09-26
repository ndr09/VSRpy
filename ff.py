import numpy as np
from matplotlib import pyplot as plt
import csv
import os

os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))
os.environ[
    'CLASSPATH'] = "C:\\Users\\Mariella\\Desktop\\mapElites\\hebbianCode\\VSRpy\\TwoDimHighlyModularSoftRobots.jar"

import jnius_config

jnius_config.add_options('-Xrs', '-Xmx4096m')

from jnius import autoclass
import jnius

String = autoclass("java.lang.String")
terrain = "hilly-3-30-0"
shape = "biped-4x3"
sensors = "high_biped-0.01-f"
controller = "HLP-full-0.01-tanh-1-1--1-1-1"
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


def loadVal(filename):
    data = dict()
    data["flat"] = list()
    data["hilly"] = list()
    data["steppy"] = list()
    # print(filename)
    with open("./" + filename, encoding='utf-8') as f:
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
    with open(filename, encoding='utf-8') as f:
        fr = csv.DictReader(f, delimiter=";")
        for l in fr:
            ser = l["best→solution→serialized"]
    return ser


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
    for i in range(10):
        tmp.append(len(data[i]))
    # print(tmp)
    for i in range(len(data[0])):
        tmp = list()
        for j in range(10):
            tmp.append(data[j][i])
        median.append(np.median(tmp))
    return median


def valCheck(filename):
    return os.path.isfile(filename)


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
        lab = ["", "MLP", "", "", "F - 0.1 - Z", "", "", "F - 0.1 - Z", "", "", "F - 0.01 - Z", "", "", "F - 0.01 - R",
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
    fig[y].set_ylabel(sensor)


def netPlot(fig, d, conf, y, sensor, mi, ma):
    lab = ["MLP", "F - 0.1 - Z", "F - 0.1 - Z", "F - 0.01 - Z", "F - 0.01 - R",
           "I - 0.1 - Z", "I - 0.1 - R", "I - 0.01 - Z", "I - 0.01 - R"]
    for x in range(len(d)):
        print(len(d[x]))
        fig[y, x].imshow(d[x], vmin=mi, vmax=ma)

    if y == 2:
        for x in range(len(lab)):
            fig[y, x].set_xlabel(lab[x])
            for tick in fig[y, x].xaxis.get_major_ticks():
                tick.label.set_fontsize('xx-small')
    fig[y, 0].set_ylabel(sensor)


def getSensors(serialized):
    observed = np.array(pyworker.locomoteSerialized(String(serialized)).getDataObservation())
    '''rf = len(observed) // len(observed[0])
    tmp = list()

    for i in range(len(observed)):
        tmp1 = list()
        for j in range(len(observed[i])):
            for k in range(rf):
                tmp1.append(observed[i][j])
        tmp.append(tmp1)

    observed = tmp
    npdata = np.array(observed)
    npdata = npdata.transpose()'''
    return observed


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
        maxs.append([np.min(data[k][i]) for i in range(10)])
    return np.min(maxs)


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
                       color='black', label="MLP")
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
                       color='black', label="MLP")
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


rng = np.random.default_rng(0)
bestSer = getBest("D:\dati hebbian/biped/high/full/001/zero/0.txt")
best = getSensors(bestSer)

print(len(best[-1]))

gsel = [rng.normal(0, 10, 1131) for i in range(1000)]
gsel1 = [rng.normal(0, 1, 1131) for i in range(1000)]
gsel2 = [rng.uniform(-10, 10, 1131) for i in range(1000)]
gsel3 = [rng.normal(-1, 1, 1131) for i in range(1000)]
ser = [[],[],[],[]]

c = 0
print("========================================")
# s = pyworker.getRobotSerialized(b)
print("----------------------------------------")
# pyworker.makeVideo(String(s), True, String(terrain),String("./test_last_"+str(c)+".mp4"),b)
print("----------------------------------------")
# pyworker.makeVideo(String(bestSer), True, String(terrain), String("./test_newversion2_norm" + str(c) + ".mp4"), list())
print("----------------------------------------")
for g in gsel:
    s = pyworker.getRobotSerialized(g.tolist())

    res = pyworker.validatationSerialized(String(s))
    ser[0].append(res[0])
    c += 1

for g in gsel1:
    s = pyworker.getRobotSerialized(g.tolist())

    res = pyworker.validatationSerialized(String(s))
    ser[1].append(res[0])
    c += 1

for g in gsel2:
    s = pyworker.getRobotSerialized(g.tolist())

    res = pyworker.validatationSerialized(String(s))
    ser[2].append(res[0])
    c += 1

for g in gsel3:
    s = pyworker.getRobotSerialized(g.tolist())

    res = pyworker.validatationSerialized(String(s))
    ser[3].append(res[0])
    c += 1

ser1 = list()

#for t in range(0, 60, 1):
    #res1 = pyworker.validatationSerializedTimed(String(bestSer), t)
    #ser1.append(res1)
    #pass

fig, axs = plt.subplots(1, 1)
axs.boxplot(ser, labels=['normal', 'normal-norm', 'uniform', 'uniform-norm'])
#axs.set_xticks([])
axs.set_ylabel("velocity")

#axs[1].boxplot(ser1)
#axs[1].set_xticklabels([])
#axs[1].set_ylabel("velocity")
plt.savefig("plots1_new3_normal_random_nonorm.png", dpi=800)

# best =np.array(best)
# plt.imshow(sum(best) / len(best))
# plt.savefig("test1.png", dpi=800)
