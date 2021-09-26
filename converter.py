import os
import csv
import numpy as np

os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))
os.environ[
    'CLASSPATH'] = "C:\\Users\\Mariella\\Desktop\\mapElites\\hebbianCode\\VSRpy\\TwoDimHighlyModularSoftRobots.jar"

import jnius_config

jnius_config.add_options('-Xrs', '-Xmx4096m')

from jnius import autoclass
import jnius


def convert(arr, bound, size):
    # a = arr.replace("[","").replace("]","").split(",")
    # print(a)
    b = list()
    for i in range(len(arr)):
        tmp = int(arr[i] // ((bound[i][0] - bound[i][1]) / size))
        if arr[i] >= bound[i][0]:
            b.append(size - 1)
        elif arr[i] <= bound[i][1]:
            b.append(0)
        else:
            b.append(tmp)
    # print(b)
    return b


def indexes(xv, yv, size):
    x = convert(xv, [[1.0, 0.0], [5.0, 0.0]], size)
    y = convert(yv, [[1.0, 0.0], [3.0, 0.0]], size)
    return x[0], x[1], y[0], y[1]


String = autoclass("java.lang.String")
terrain = "hilly-1-30-0"
shape = "biped-4x3"
sensors = "high_biped-0.01-f"
controller = "MLP-0.1-0-tanh"
duration = 60
Pyworker = autoclass('it.units.erallab.hmsrobots.Pyworker')
pyworker = Pyworker(String(terrain), String(shape), String(sensors),
                    String(controller), duration)
first_serialized = ''
last_serialized = ''
sizem = 10

for iter in range(23679, 23680, 2631):
    print(iter)
    heatmap_fitness = np.full((sizem, sizem, sizem, sizem), np.NaN, dtype=np.float)
    heatmap_gen = list()
    for i in range(sizem):
        heatmap_gen.append(list())
        for k in range(sizem):
            heatmap_gen[i].append(list())
            for j in range(sizem):
                heatmap_gen[i][k].append(list())
                for l in range(sizem):
                    heatmap_gen[i][k][j].append(list())

    with open("gait/gait_0_high_biped_" + str(iter) + ".csv") as f:
        fr = csv.DictReader(f)
        best = 0
        best_genome = []
        for r in fr:
            x1, x2, y1, y2 = indexes([float(r["behavior_0"]), float(r["behavior_1"])],
                                     [float(r["behavior_2"]), float(r["behavior_3"])], sizem)
            heatmap_fitness[x1, y1, x2, y2] = float(r["objective"])

            tmp = list()
            for j in range(300):
                tmp.append(float(r["solution_" + str(j)]))
            if not tmp:
                print(x1)
                print(x2)
                print(y1)
                print(y2)
            heatmap_gen[x1][y1][x2][y2] = tmp
    first = None
    fp = None
    lp = None
    last = None
    for i in range(sizem):
        for j in range(sizem):
            for k in range(sizem):
                for l in range(sizem):
                    if (not np.isnan(heatmap_fitness[i, j, k, l])) and heatmap_fitness[i, j, k, l] > 3.0:
                        if first is None:
                            first = heatmap_gen[i][j][k][l]
                            fp = (i, j, k, l)
                        last = heatmap_gen[i][j][k][l]
                        lp = (i, j, k, l)

    if first is not None:
        first_serialized = pyworker.getRobotSerialized(first)
    if last is not None:
        last_serialized = pyworker.getRobotSerialized(last)
    print(fp)
    print(lp)
    print("--------------")
    ll = list()
    ll.append(heatmap_gen[0][0][0][0])
    ll.append(heatmap_gen[0][9][0][9])
    ll.append(heatmap_gen[1][9][0][9])
    ll.append(heatmap_gen[1][9][1][9])
    ll.append(last)
    print(ll)
    lls = [pyworker.getRobotSerialized(jj) for jj in ll]

    with open("gait/gait_video_" + str(iter), "w") as f:
        f.write("x;y;best.serialized.robot\n")
        c = 0
        for jjs in lls:
            f.write("x;" + str(c) + ";" + jjs + "\n")
            c += 1
