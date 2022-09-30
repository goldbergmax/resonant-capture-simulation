import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from disk_models import StochasticDisk
from capture_simulation import CaptureSimulation

def change_param(i):
    disk = StochasticDisk(fixed_tau_a=fixed_tau_a, fixed_tau_e=fixed_tau_a/K, kappa=kappas[i])
    sim = CaptureSimulation(p, q, m1, m2, a_init, Delta_init, disk)
    sim.run_sim(disk_end, t_end, t_start_removal=t_start_removal, write_dir=path / f'{i}.pkl')
    return sim.get_results_row()

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=3)
parser.add_argument('--K', type=float, default=300)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--adiabatic', action=argparse.BooleanOptionalAction, default=True)
parser.add_argument('--dirname', type=str, default='')
args = parser.parse_args()
p, K, N = args.p, args.K, args.N
dirname = args.dirname
if dirname:
    dirname = '_' + dirname

q = 1
m1, m2 = 10/332900, 10/332900

a_init = 0.1
Delta_init = 0.1
fixed_tau_a = -3e5

disk_end = 3e4
t_end = 4e4
t_start_removal = 0.75 if args.adiabatic else 1
adiabatic_str = 'adiabatic' if args.adiabatic else 'nonadiabatic'

kappas = np.logspace(-7, -5, N)

path = Path(f'results/stoch_disk/{p}_{q}_{adiabatic_str}_{K}{dirname}')
os.makedirs(path, exist_ok=True)

pool = Pool(50)
df = pd.DataFrame(pool.map(change_param, range(N)))
df.to_csv(path / f'results.csv')
