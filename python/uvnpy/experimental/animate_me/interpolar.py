import numpy as np
from scipy.interpolate import interp1d

# archivo = 'c_uopt.csv' # tiene mas columnas que el resto, es cluster
# archivo = 'r1cmd.csv' # comandos generados despues del Jinverso
# archivo = 'r2cmd.csv' # idem
# archivo = 'uav1_cmdvel.csv' # comandos que salen del robot_adapter
archivo = 'uav2_cmdvel.csv'    # comandos que salen del robot_adapter
datum = np.loadtxt(archivo, delimiter=',')
keepme = np.diff(datum[:, 0]) != 0
# interp1d no admite tiempos repetidos
if not np.all(keepme):
    datum = datum[np.insert(keepme, 0, True), ...]
f = interp1d(
    datum[:, 0], datum[:, 1:],
    axis=0, kind='cubic')    # kind puede ser 'zero'

t = 300
vx, vy, vz, wz = f(t)
print(vx, vy, vz, wz)
