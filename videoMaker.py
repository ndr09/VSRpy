import numpy as np
from matplotlib import pyplot as plt
import csv
import os
from pprint import pprint as pp

os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))
os.environ[
    'CLASSPATH'] = "C:\\Users\\Mariella\\Desktop\\mapElites\\hebbianCode\\VSRpy\\VSREO.jar"

import jnius_config

jnius_config.add_options('-Xrs', '-Xmx4096m')

from jnius import autoclass
import jnius

String = autoclass("java.lang.String")
terrain = "hilly-3-30-0"
duration = 60
VideoMaker = autoclass('it.units.erallab.VideoMaker')


def getBest(filename):
    ser = ''
    with open( filename, encoding='utf-8') as f:
        fr = csv.DictReader(f, delimiter=";")
        for l in fr:
            ser = l["best→solution→serialized"]
    return ser


def writeData(bests):
    with open("tmp.txt", "w") as f:
        f.write("x;y;best.serialized.robot\n")
        for i in range(8):
            f.write(str(i % 4) + ";" + str(i // 4) + ";" + bests[i] + "\n")
        f.write(str(1) + ";" + str(2) + ";" + bests[8] + "\n")
        f.write(str(2) + ";" + str(2) + ";" + bests[9] + "\n")


def makeVideo(bests, filename):
    breakables = ["identity"]
    writeData(bests)
    inputFile = 'inputFile=tmp.txt'
    outputFile = 'outputFile=' + "video/norm/" + filename.replace("/", "_")
    print(len(bests))
    for trans in breakables:
        tt = "transformation=" + trans
        of = outputFile + "_" + trans + ".mp4"
        args = [inputFile, of, tt]
        print(trans)
        VideoMaker.main(args)


if __name__ == '__main__':
    shapes = ['worm']
    sensors = ['low']
    models = ['full']
    inits = ['zero']
    etas = ['001']
    baseDir = 'D:/dati hebbian/rnorm'

    ''' best = list()
    for i in range(10):
        best.append(getBest("results/biped/high/full/001/random/" + str(i) + ".txt"))
    makeVideo(best, "biped/high/full/001/random")'''

    for shape in shapes:
        for sensor in sensors:
            tmp = list()
            print("make video " + shape + "/" + sensor)
            for model in models:
                for eta in etas:
                    for init in inits:
                        tmp = list()
                        for i in range(10):
                            tmp.append(getBest(
                                baseDir + "/" + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init + "/nb_" + str(
                                    i) + ".txt"))
                        print("make video " + shape + "/" + sensor + "/" + model + "/" + eta + "/" + init)
                        makeVideo(tmp, "nb_"+shape + "/" + sensor + "/" + model + "/" + eta + "/" + init)
