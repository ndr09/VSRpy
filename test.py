import os
import numpy as np
#--terrain=hilly-1-30-0 --shape=biped-4x3 --sensors=high_biped-0.01-f --controller=MLP-0.1-0-tanh --duration=60 --births=20 --seed=0
from ribs.archives import GridArchive
from ribs.emitters import ImprovementEmitter
from ribs.optimizers import Optimizer
from multiprocessing import Pool
from ribs.visualize import grid_archive_heatmap

os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))
#os.environ[
#    'CLASSPATH'] = "/home/andrea.ferigo/pyVsr/TwoDimHighlyModularSoftRobots.jar"

import jnius_config
import argparse

jnius_config.add_options('-Xmx4096m')
jnius_config.add_classpath("C:\\Users\\Mariella\\Desktop\\mapElites\\hebbianCode\\VSREpy\\TwoDimHighlyModularSoftRobots.jar")

from jnius import autoclass
import jnius


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
    parser.add_argument("--n_parallel", help="number of core to use", type=int, default=19)
    parser.add_argument("--savefile", help="where save the information", default='pyme.txt')
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
    print(args)
    process_pool = Pool(n_parallel)
    print("pool creation")
    from functools import partial
    # multiprocessing pool imap function accepts only one argument at a time, create partial function with
    # constant parameters
    print(params)
    f = partial(fitness_function, params)
    fitness = list(process_pool.imap(f, candidates))
    process_pool.close()
    return fitness


def locomotion(args, ind):
    print("loco")
    #return [1.0, [0.1,0.2,0.3,0.4]]
    try:
        print(args)
        print(jnius_config.get_classpath())
        String = autoclass("java.lang.String")
        Pyworker = autoclass('it.units.erallab.hmsrobots.Pyworker')
        Outcome = autoclass('it.units.erallab.hmsrobots.tasks.locomotion.Outcome')
        #jnius_config.add_classpath("/home/andrea.ferigo/pyVsr/TwoDimHighlyModularSoftRobots.jar")
        pyworker = Pyworker(String(args[0]), String(args[1]), String(args[2]),
                        String(args[3]), args[4], ind.tolist())
        print("b")
        print(pyworker.locomote)
        s = pyworker.getRobotSerialized()
        print(s)        
        outcome = pyworker.locomote()
        print("c")
        tmp = outcome.getSpectrumDescriptor()
        return [outcome.getVelocity(), [tmp[0], tmp[2], tmp[1], tmp[3]]]
        #return [1.0, [0.1,0.2,0.3,0.4]]
    finally:
        jnius.detach()

#String = autoclass("java.lang.String")
#Pyworker = autoclass('it.units.erallab.hmsrobots.Pyworker')

if __name__ == '__main__':
    args = input_parser()
    terrain = args.terrain
    shape = args.shape
    sensors = args.sensors
    controller = args.controller
    duration = args.duration
    seed = args.seed
    births = args.births
    genome_length = args.genome_length
    n_parallel = args.n_parallel
    filename = args.savefile
   
    archive = GridArchive([20, 20, 20, 20], [(0, 6), (0, 7), (0, 1), (0, 1)])
    emitters = [ImprovementEmitter(archive, [0.0] * genome_length, 0.1, batch_size=2, seed=seed)]
    optimizer = Optimizer(archive, emitters)
    total_births = 0
    generations = (births // 19) + 1
    savepoints = [i * (generations // 10) for i in range(0, 10)]
    epoch = 0
    print("a")
    while total_births < births:
        solutions = optimizer.ask()
        total_births += len(solutions)
        print(len(solutions))
        fits = list()
        bcs = list()
        results =list()
        #for s in solutions:
        #    results.append(locomotion([terrain, shape, sensors, controller, duration], s))
        results = multiprocess_function(solutions, {"n_parallel": n_parallel, "fitness_function": locomotion,
                                                    "params": [terrain, shape, sensors, controller, duration]})
        for res in results:
            fits.append(res[0])
            bcs.append(res[1])

        optimizer.tell(fits, bcs)
        if epoch in savepoints:
            savedata(str(seed) + "_" + sensors.split('-')[0] + "_" + str(epoch) + ".csv",
                     archive.as_pandas(include_solutions=True))
        epoch += 1
    savedata(str(seed) + "_" + sensors.split('-')[0] + "_final.csv",
             archive.as_pandas(include_solutions=True))
# python_instance_of_the_Java_class = TheJavaClass()
