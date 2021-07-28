import os
# queste due righe sono da commentare su linux (mi pare, not sure)
os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))
os.environ[
    'CLASSPATH'] = "C:\\Users\\Mariella\\Desktop\\mapElites\\hebbianCode\\VSRpy\\TwoDimHighlyModularSoftRobots.jar"

import jnius_config
import argparse
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from multiprocessing import Pool
from matplotlib import pyplot as plt
from numpy.ma import masked_array

jnius_config.add_options('-Xrs', '-Xmx4096m')

from jnius import autoclass
import jnius
import numpy as np


def input_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--terrain", help="type of terrain i.e. hilly-1-30-0")
    parser.add_argument("--shape", help="body shape i.e. biped-4x3")
    parser.add_argument("--sensors", help="sensors equipped i.e.high_biped-0.01-f")
    parser.add_argument("--controller", help="type of controller i.e. MLP-0.1-0-tanh")
    parser.add_argument("--duration", help="duration of locomotion", type=float)
    parser.add_argument("--births", help="total number of births", type=int)
    parser.add_argument("--seed", help="seed", type=int)
    parser.add_argument("--genome_length", help="number of variable", type=int, default=300)
    parser.add_argument("--n_parallel", help="number of core to use", type=int, default=10)
    parser.add_argument("--savefile", help="where prefix of the savefile", default='test')
    args = parser.parse_args()
    return args


def savedata(namefile, data):
    data.to_csv(namefile)


def multiprocess_function(candidates, args):
    """
    evaluation of candidates' fitness using multiprocessing pool
    """
    n_parallel = args["n_parallel"]
    fitness_function = args["fitness_function"]
    params = args["params"]
    # -------------- multiprocessing ----------------
    process_pool = Pool(n_parallel)

    from functools import partial
    # multiprocessing pool imap function accepts only one argument at a time, create partial function with
    # constant parameters
    f = partial(fitness_function, params)
    fitness = list(process_pool.imap(f, candidates))
    process_pool.close()
    return fitness


def locomotion(args, ind):
    String = autoclass("java.lang.String")
    Pyworker = autoclass('it.units.erallab.hmsrobots.Pyworker')
    pyworker = Pyworker(String(args[0]), String(args[1]), String(args[2]),
                        String(args[3]), args[4])
    outcome = pyworker.locomoteSerialized(String(ind))
    tmp = outcome.getDataObservation()
    return tmp

def drawImage(observed, filename, dpi=400):

    print(len(observed))
    print(len(observed[0]))
    rf = len(observed)//len(observed[0])
    tmp = list()

    for i in range(len(observed)):
        tmp1 = list()
        for j in range(len(observed[i])):
            for k in range(rf):
                tmp1.append(observed[i][j])
        tmp.append(tmp1)

    observed = tmp
    print(len(observed))
    print(len(observed[0]))
    print(observed[0])
    npdata = np.array(observed)
    npdata = npdata.transpose()
    plt.imshow(npdata, origin='lower')
    plt.yticks([i for i in range(0, len(observed[0]), 4*rf)], [str(i//(4*rf)) for i in range(0, len(observed[0]),4*rf)])
    plt.savefig(filename, dpi=dpi)

if __name__ == '__main__':
    args = input_parser()
    terrain = args.terrain
    shape = args.shape
    sensors = args.sensors
    controller = args.controller
    filename = args.savefile
    duration = args.duration

    String = autoclass("java.lang.String")
    Pyworker = autoclass('it.units.erallab.hmsrobots.Pyworker')

    with open("data/data.txt") as f:
        for robot in f:
            observed = locomotion([String(terrain), String(shape), String(sensors), String(controller), duration],
                                  robot)
            drawImage(observed[:190], "data/data2.pdf", 600)
