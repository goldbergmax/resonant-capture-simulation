import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from disk_models import BasicEccentricDisk
from capture_simulation import CaptureSimulation

def change_param(i):
    disk = BasicEccentricDisk(e0=e0, q=q_disk, fixed_tau_a=fixed_tau_e*Ks[i], fixed_tau_e=fixed_tau_e, aout=aout)
    sim = CaptureSimulation(p, q, m1, m2, a_init, Delta_init, disk)
    disk_end = 3*sim.get_estimated_migration_finish()
    t_end = disk_end + 1e4
    sim.run_sim(disk_end, t_end, t_start_removal=t_start_removal, write_dir=path / f'{p}_{q}_results_{adiabatic_str}_{e0}_{i}.pkl')
    return sim.get_results_row()

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=3)
parser.add_argument('--N', type=int, default=96)
parser.add_argument('--adiabatic', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--e0', type=float, default=0.1)
args = parser.parse_args()
p, N, e0 = args.p, args.N, args.e0

q = 1
m1, m2 = 10/332900, 10/332900
a_init = 0.1
Delta_init = 0.1

fixed_tau_e = -3e2
q_disk = 0
aout = 0.1

t_start_removal = 0.75 if args.adiabatic else 1
adiabatic_str = 'adiabatic' if args.adiabatic else 'nonadiabatic'

Ks = np.logspace(np.log10(1e2), np.log10(1e5), N)
if e0 > 0:
    path = Path(f'results/eccentric_disk')
else:
    path = Path(f'results/regular_disk')
os.makedirs(path, exist_ok=True)

pool = Pool(48)
df = pd.DataFrame(pool.map(change_param, range(N)))
df.to_csv(path / f'{p}_{q}_results_{adiabatic_str}_{e0}.csv')