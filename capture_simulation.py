from __future__ import annotations
import warnings
import numpy as np
import matplotlib.pyplot as plt
import rebound, reboundx
from pathlib import Path
import pickle
from hammy import Delta_eq, get_f_g

class CaptureSimulation():
    def __init__(self, p: int, q: int, m1: float, m2: float, a1_init: float, 
                 Delta_init: float, disk, perturber=False):
        """
        Initialize a simulation of MMR capture with a given disk

        Parameters
        --------
        p, q : int
            Resonant index and order, for capture into p:p-q resonance
        m1, m2 : float
            Planet masses in Solar Masses
        a1_init : float
            Initial semi-major axis of inner planet in AU
        Delta_init : float
            Initial spacing from resonance
        disk : EccentricDisk
        perturber : bool
            If True, add an outer perturbing giant planet
        """
        self.p, self.q = p, q
        self.f, self.g = get_f_g(self.p, self.q)
        self.m1, self.m2 = m1, m2
        self.a1_init, self.Delta_init = a1_init, Delta_init
        self.sim = rebound.Simulation()
        self.sim.units = ('yr', 'Msun', 'AU')
        self.sim.add(m=1)
        self.sim.add(m=m1, a=a1_init, e=0.01, pomega=0.)
        a2_init = ((Delta_init+1)*self.p/(self.p-self.q))**(2/3)*a1_init
        self.sim.add(m=m2, a=a2_init, e=0.01, pomega=0.)
        if perturber:
            self.sim.add(m=1e-3, a=0.8, e=0.3, pomega='uniform')
        self.ps = self.sim.particles
        self.sim.integrator = 'whfast'
        self.sim.dt = self.ps[1].P/20
        rebx = reboundx.Extras(self.sim)
        mof = rebx.load_operator("modify_orbits_direct")
        sto = rebx.load_force("stochastic_forces")
        rebx.add_operator(mof)
        rebx.add_force(sto)
        self.disk = disk
    
    def run_sim(self, disk_end: float, t_end: float, write_dir: str=None, 
                tidal_tau_e: float=np.inf, e_foldings: int=5, t_start_removal=0.75) -> None:
        """
        Run a simulation of MMR capture

        Parameters
        --------
        disk_end : float
            Time of disk removal
        t_end : float
            Simulation end time in yr
        write_dir : str
            If not None, write hdf5 with simulation results to that directory 
            with the filename the time of initializion
        tidal_tau_e : float
            If not np.inf, use this value for the tidal eccentricity damping timescale after the disk is removed
        e_foldings : int
            Number of e-foldings to use in the disk removal
        t_start_removal : float
            Time after which to start removing the disk, in fraction of disk_end
        """
        self.disk_end = disk_end
        self.t_end = t_end
        self.t_start_removal = t_start_removal
        assert disk_end <= t_end, "Disk removal time must be less than simulation end time"
        self.generate_warnings(disk_end, t_start_removal)
        self.times = np.arange(0.,t_end,1)
        Nout = len(self.times)
        self.orbits = np.zeros((Nout, 2, 4))
        for i,t in enumerate(self.times):
            self.orbits[i,0] = self.ps[1].a, self.ps[1].e, self.ps[1].pomega, self.ps[1].l
            self.orbits[i,1] = self.ps[2].a, self.ps[2].e, self.ps[2].pomega, self.ps[2].l
            scaling_factor = get_scaling_factor(t, disk_end, e_foldings=e_foldings, t_start_removal=t_start_removal)
            self.disk.apply_forces(self.ps[1], scaling_factor)
            self.disk.apply_forces(self.ps[2], scaling_factor)
            if t >= disk_end:
                for par in self.ps[1:3]:
                    par.params['tau_e'] = tidal_tau_e
            self.sim.integrate(t)
        if write_dir:
            path = Path(write_dir)
            with open(path, 'wb') as f:
                pickle.dump(self, f)

    def __getstate__(self):
        """
        Return a dictionary of the object's state with unpickleable simulation removed
        """
        state = self.__dict__.copy()
        del state['sim'], state['ps']
        return state

    def get_estimated_migration_finish(self) -> float:
        """
        Estimate the time at which the pair will reach exact commensurability
        """
        return 2/3*self.Delta_init*-self.disk.tau_a(self.ps[2])

    def get_estimated_lib_time(self) -> float:
        """
        Estimate the libration timescale in resonance
        """
        return ((self.ps[1].m + self.ps[2].m)/self.ps[0].m)**(-2/3) * self.ps[1].P

    def generate_warnings(self, disk_end, t_start_removal) -> None:
        """
        Generate warnings for the simulation
        """
        estimated_migration_finish = self.get_estimated_migration_finish()
        if estimated_migration_finish > 0.8*t_start_removal*disk_end:
            warnings.warn(f"Migration may not finish (est. {estimated_migration_finish:.2f}) before disk removal begins ({t_start_removal*disk_end:.2f})")
        
        estimated_lib_time = self.get_estimated_lib_time()
        if estimated_lib_time > 0.5*np.abs(self.disk.tau_e(self.ps[1])):
            warnings.warn(f'Damping of {self.disk.tau_e(self.ps[1]):.3f} may not be adiabatic (est. libration period {estimated_lib_time:.3f})')
    
    def per_rat(self) -> np.ndarray:
        """Compute the period ratio of the planets"""
        return (self.orbits[:,1,0]/self.orbits[:,0,0])**1.5
    
    def Delta(self) -> np.ndarray:
        """Get distance from resonance"""
        return self.per_rat()*(self.p - self.q)/self.p - 1

    def res_angles(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute both resonant angles"""
        phi_1 = np.remainder(self.p*self.orbits[:,1,3] - (self.p-self.q)*self.orbits[:,0,3] - self.q*self.orbits[:,0,2] + np.pi, 2*np.pi) - np.pi
        phi_2 = np.remainder(self.p*self.orbits[:,1,3] - (self.p-self.q)*self.orbits[:,0,3] - self.q*self.orbits[:,1,2], 2*np.pi)
        return phi_1, phi_2

    def get_theta_hat(self) -> np.ndarray:
        """Compute the combined resonant angle"""
        phi_1, phi_2 = self.res_angles()
        return np.remainder(np.arctan2(self.f*self.orbits[:,0,1]*np.sin(phi_1) + self.g*self.orbits[:,1,1]*np.sin(phi_2), 
                                       self.f*self.orbits[:,0,1]*np.cos(phi_1) + self.g*self.orbits[:,1,1]*np.cos(phi_2)), 2*np.pi)

    def get_complex_Z(self) -> np.ndarray:
        """Compute the complex free eccentricity normalized by the f and g coefficients"""
        z1 = self.orbits[:,0,1]*np.exp(1j*self.orbits[:,0,2])
        z2 = self.orbits[:,1,1]*np.exp(1j*self.orbits[:,1,2])
        Z_complex = (self.f*z1 + self.g*z2)/np.sqrt(self.f**2 + self.g**2)
        return Z_complex

    def get_abs_Z(self) -> np.ndarray:
        """Compute the absolute value of the complex normalized free eccentricity"""
        return np.abs(self.get_complex_Z())

    def get_ecc_vectors(self) -> tuple[np.ndarray, np.ndarray]:
        """Compute eccentricity vectors"""
        return self.orbits[:,:,1]*np.cos(self.orbits[:,:,2]), self.orbits[:,:,1]*np.sin(self.orbits[:,:,2])

    def get_basic_results(self) -> dict:
        """Get basic results of the simulation as a dictionary"""
        phi_1, phi_2 = self.res_angles()
        theta_hat = self.get_theta_hat()
        return {'Delta':self.Delta()[-1], 
                'e1':self.orbits[-1,0,1], 
                'e2':self.orbits[-1,1,1], 
                'theta_hat':theta_hat[-1],
                'phi1_librating':is_librating(phi_1[9*len(self.orbits)//10:]), 
                'phi2_librating':is_librating(phi_2[9*len(self.orbits)//10:]),
                'theta_hat_librating':is_librating(theta_hat[9*len(self.orbits)//10:]),
                'Delta_pomega':np.diff(self.orbits[-1,:,2])[0], 
                'Zabs':self.get_abs_Z()[-1]}

    def get_results_row(self) -> dict:
        """Get a row of initial conditions and results for the dataframe"""
        ret_dict = {'p':self.p, 'q':self.q, 'm1':self.m1, 'm2':self.m2, 
                    'a1_init':self.a1_init, 'Delta_init':self.Delta_init, 
                    'disk_end':self.disk_end, 't_end':self.t_end, 't_start_removal':self.t_start_removal}
        ret_dict |= self.disk.describe()
        ret_dict |= self.get_basic_results()
        return ret_dict

    def plot(self):
        fig, axs = plt.subplots(2, 3, figsize=(18,10), sharex=True, gridspec_kw={'hspace':0.05, 'wspace':0.3})
        axs[0,0].plot(self.times, self.orbits[:,:,0])
        axs[0,0].set_ylim(0, self.a1_init*2)
        axs[0,0].set_ylabel('Semi-major axis (AU)', fontsize=14)

        axs[0,1].plot(self.times, self.per_rat())
        axs[0,1].axhline(self.p/(self.p-1), c='k', linestyle='-')
        axs[0,1].axhline(self.p/(self.p-1)*(1+Delta_eq(self.m1, self.m2, self.p, -self.disk.fixed_tau_a, self.disk.fixed_tau_e)), c='r', linestyle='-')
        axs[0,1].set_ylim(min(0.99*self.p/(self.p-1), 0.99*self.per_rat().min()), None)
        axs[0,1].set_ylabel('$P_2/P_1$', fontsize=14)
        axs[0,1].text(0.5, 0.5, f'$\Delta={self.per_rat()[-1]*(self.p-1)/self.p-1:.4f}$', transform=axs[0,1].transAxes, fontsize=16)

        axs[0,2].scatter(self.times, self.get_theta_hat(), s=1)
        axs[0,2].set_ylim(0, 2*np.pi)
        axs[0,2].set_ylabel(r'$\hat{\theta}$', fontsize=14)

        axs[1,0].scatter(self.times, self.orbits[:,0,2], s=0.1)
        axs[1,0].scatter(self.times, self.orbits[:,1,2], s=0.1)
        axs[1,0].set_ylim(-np.pi, np.pi)
        axs[1,0].set_xlabel('Time', fontsize=14)
        axs[1,0].set_ylabel(r'$\varpi$', fontsize=14)

        phi_1, phi_2 = self.res_angles()
        axs[1,1].scatter(self.times, phi_1, s=0.5, alpha=0.5)
        axs[1,1].scatter(self.times, phi_2, s=0.5, alpha=0.5)
        axs[1,1].set_ylim(-np.pi, 2*np.pi)
        axs[1,1].set_xlabel('Time', fontsize=14)
        axs[1,1].set_ylabel('Resonant angles', fontsize=14)

        axs[1,2].scatter(self.times, self.orbits[:,0,1], s=1)
        axs[1,2].scatter(self.times, self.orbits[:,1,1], s=1)
        axs[1,2].set_xlabel('Time', fontsize=14)
        axs[1,2].set_ylabel('$e$', fontsize=14)

        for ax in axs.flatten():
            ax.axvline(self.disk_end, linestyle='--', color='k')
            ax.axvline(self.t_start_removal*self.disk_end, linestyle='--', color='k')
        return fig, axs

def get_scaling_factor(t: float, disk_end: float, e_foldings=5, t_start_removal=0.75):
    """
    Scaling factor to remove the disk

    Parameters
    ----------
    t : float
        time
    disk_end : float
        Time at which the disk ends
    e_foldings : int
        Number of e-foldings during removal
    t_start_removal : float
        Time at which to start removing the disk
    """
    assert 0 <= t_start_removal <= 1
    scaling_factor = 1
    if t_start_removal*disk_end <= t < disk_end:
        scaling_factor = np.exp(e_foldings*(t - t_start_removal*disk_end)/((1 - t_start_removal)*disk_end))
    elif t >= disk_end:
        scaling_factor = np.inf
    return scaling_factor

def is_librating(angle: np.ndarray) -> bool:
    return np.max(angle) - np.min(angle) < 3*np.pi/2
