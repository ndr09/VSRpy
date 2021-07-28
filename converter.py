import os
import csv

os.add_dll_directory(os.path.join(os.environ['JAVA_HOME'], 'bin', 'server'))
os.environ[
    'CLASSPATH'] = "C:\\Users\\Mariella\\Desktop\\mapElites\\hebbianCode\\VSREpy\\TwoDimHighlyModularSoftRobots.jar"

import jnius_config

jnius_config.add_options('-Xrs', '-Xmx4096m')

from jnius import autoclass
import jnius

String = autoclass("java.lang.String")
terrain = "hilly-1-30-0"
shape = "biped-4x3"
sensors = "high_biped-0.01-f"
controller = "MLP-0.1-0-tanh"
duration = 60
Pyworker = autoclass('it.units.erallab.hmsrobots.Pyworker')
pyworker = Pyworker(String(terrain), String(shape), String(sensors),
                    String(controller), duration)
best_serialized = ''
for i in range(0,5261,526):
    with open("../terrains/hilly3/hilly3_0_high_biped_"+str(i)+".csv") as f:
        fr = csv.DictReader(f)
        best = 0
        best_genome = []
        for r in fr:
            if float(r["objective"]) >= best:
                best = float(r["objective"])
                tmp = []
                for j in range(300):
                    tmp.append(float(r["solution_"+str(j)]))
                best_genome = tmp
        print(best)
        best_serialized = pyworker.getRobotSerialized(best_genome)

    with open("../terrains/hilly3/video_"+str(i), "w") as f:
        f.write("x;y;best.serialized.robot\n")
        f.write("0;0;"+best_serialized+"\n")

