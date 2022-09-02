import os, argparse
from pathlib import Path
from itertools import product
import numpy as np
import pandas as pd
from multiprocessing import Pool
from disk_models import BasicEccentricDisk
from capture_simulation import CaptureSimulation

def change_param(params):
    Delta_init, tidal_tau_e = params
    disk = BasicEccentricDisk(e0=0., q=0, fixed_tau_a=-np.inf, fixed_tau_e=-np.inf)
    sim = CaptureSimulation(p, q, m1, m2, a_init, Delta_init, disk)
    sim.run_sim(0, t_end, tidal_tau_e=tidal_tau_e, write_dir=path / f'{p}_{q}_results_{Delta_init}_{tidal_tau_e}.pkl')
    return sim.get_results_row()

parser = argparse.ArgumentParser()
parser.add_argument('--p', type=int, default=3)
args = parser.parse_args()

p = args.p
q = 1
m1, m2 = 10/332900, 10/332900

a_init = 0.1
t_end = 1e5
Delta_inits = np.linspace(-0.05, 0.05, 20)
tidal_tau_es = [-1e3, -1e4, -1e5]

path = Path(f'results/tides')
os.makedirs(path, exist_ok=True)

pool = Pool(48)
df = pd.DataFrame(pool.map(change_param, product(Delta_inits, tidal_tau_es)))
df.to_csv(path / f'{p}_{q}_results.csv')