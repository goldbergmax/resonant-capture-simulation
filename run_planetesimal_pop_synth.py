import os, argparse
from pathlib import Path
import numpy as np
import pandas as pd
from multiprocessing import Pool
from disk_models import BasicCircularDisk, PlanetesimalDisk
from capture_simulation import CaptureSimulation

def change_param(i):
    m1, m2 = 10/332900, 10/332900
    disk = BasicCircularDisk(fixed_tau_e, fixed_tau_a)
    pl_disk = PlanetesimalDisk(N_pts=N_pts, disk_mass=disk_masses[i], loc=loc, dyn_T=dyn_T)
    sim = CaptureSimulation(p, q, m1, m2, a1_init, Delta_init, disk, pl_disk)
    sim.run_sim(disk_end, t_end, write_dir=path / f'{i}.pkl')
    return sim.get_results_row()

parser = argparse.ArgumentParser()
parser.add_argument('path', type=str, help='Path to write results to')
parser.add_argument('--p', type=int, default=3)
parser.add_argument('--K', type=float, default=300)
parser.add_argument('--N', type=int, default=100)
parser.add_argument('--T', type=float, default=0.01)
parser.add_argument('--adiabatic', action=argparse.BooleanOptionalAction, default=True)
group = parser.add_mutually_exclusive_group()
group.add_argument('--around', action=argparse.BooleanOptionalAction, default=False)
group.add_argument('--inside', action=argparse.BooleanOptionalAction, default=False)
group.add_argument('--outside', action=argparse.BooleanOptionalAction, default=False)
args = parser.parse_args()
p, K, N, dyn_T = args.p, args.K, args.N, args.T

if args.around:
    loc = 'around'
elif args.inside:
    loc = 'inside'
elif args.outside:
    loc = 'outside'
else:
    raise ValueError('Must specify location of planetesimal disk')

q = 1
m1, m2 = 10/332900, 10/332900

a1_init = 0.1
Delta_init = 0.01
fixed_tau_e = -1e2
fixed_tau_a = K*fixed_tau_e
N_pts = 1000

disk_end = 3e2
t_end = 5e4
t_start_removal = 0.75 if args.adiabatic else 1
adiabatic_str = 'adiabatic' if args.adiabatic else 'nonadiabatic'

disk_masses = np.logspace(-1, 1, N)/332900

path = Path(args.path) / Path(f'results/planetesimals/{p}_{q}_{loc}_{dyn_T}')
os.makedirs(path, exist_ok=True)

pool = Pool(50)
df = pd.DataFrame(pool.map(change_param, range(N)))
df.to_csv(path / 'results.csv')
