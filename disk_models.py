from __future__ import annotations
import numpy as np
import pandas as pd
import rebound

class EccentricDisk():
    """Class to compute planet-disk interactions in an eccentric disk"""
    def __init__(self, Sigma0: float, p: float, e0: float, q: float, h_over_r: float, Mstar=1.,
                    pomega=0., aout=1., beta_T=3/7, G=39.42):
        """
        Parameters
        --------
        Sigma0 : float
            Surface density at reference radius
        p : float
            Power law index of surface density
        e0 : float
            Eccentricity at reference radius
        q : float
            Power law index of eccentricity
        h_over_r : float
            Disk aspect ratio, assumed to be constant with radius
        Mstar : float
            Mass of the star, default 1Msun
        pomega : float
            Disk longitude of pericenter, assumed to be constant with radius
        aout : float
            Reference radius of disk
        beta_T : float
            Power law index of temperature
        G : float
            Gravitational constant, default is in AU^3 yr^-2 Msun^-1
        """
        self.aout = aout
        self.Sigma0, self.p = Sigma0, p
        self.e0, self.q, self.pomega = e0, q, pomega
        self.h_over_r, self.beta_T = h_over_r, beta_T
        self.G, self.Mstar = G, Mstar
    
    def Sigma(self, r: float) -> float:
        """Get the disk surface density at r"""
        return self.Sigma0*(self.aout/r)**self.p
    
    def e_d(self, r: float) -> float:
        """Get the disk eccentricity at r"""
        return self.e0*(self.aout/r)**self.q
    
    def A_d(self, p: rebound.Particle) -> float:
        """Disk disturbing function coefficient A_d from Silsbee & Rafikov (2015)"""
        # phi_1 is a slowly varying function of p and q and alpha
        phi_1 = -0.5
        return 2*np.pi*self.G*self.Sigma(p.a)*phi_1/(p.a*p.n)
    
    def B_d(self, p: rebound.Particle) -> float:
        """Disk disturbing function coefficient B_d from Silsbee & Rafikov (2015)"""
        # phi_2 is a slowly varying function of p and q and alpha
        phi_2 = 1.5
        return np.pi*self.G*self.Sigma(p.a)*self.e_d(p.a)*phi_2/(p.a*p.n)

    def tau_e(self, p: rebound.Particle) -> float:
        """Normal eccentricity damping timescale, e.g. Cresswell & Nelson (2008)"""
        # ignoring order unity corrections
        return -self.Mstar**2/(p.m*self.Sigma(p.a)*p.a**2)*self.h_over_r**4/p.n
    
    def tau_a(self, p: rebound.Particle) -> float:
        """Normal migration timescale"""
        # ignoring order unity corrections
        return -self.Mstar**2*self.h_over_r**2/(p.m*self.Sigma(p.a)*p.a**2*p.n)
    
    def __repr__(self) -> str:
        # make a placeholder simulation with a 10M_E particle at 1AU to compute the disk properties
        sim = rebound.Simulation()
        sim.units = ('yr', 'Msun', 'AU')
        sim.add(m=1)
        sim.add(m=3e-5, a=1)
        ret_val = ''
        ret_val += f'Precession timescale: {2*np.pi/self.A_d(sim.particles[1]):.4f}\n'
        ret_val += f'Equilibrium eccentricity: {-self.B_d(sim.particles[1])/self.A_d(sim.particles[1]):.4f}\n'
        ret_val += f'e-damping timescale: {self.tau_e(sim.particles[1]):.4f}\n'
        ret_val += f't_a/t_e = {self.tau_a(sim.particles[1])/self.tau_e(sim.particles[1]):.4f}'
        return ret_val

    def describe(self) -> dict:
        return {'Sigma0':self.disk.Sigma0, 'disk_p':self.disk.p, 'e0':self.disk.e0, 'disk_q':self.disk.q, 'h_over_r':self.disk.h_over_r}

    def disk_e_damping(self, p: rebound.Particle, e_disk_eff=None, pomega_disk=None, prec_A=None) -> tuple[float,float]:
        """
        Compute eccentricity derivative and precession timescale for a particle in the disk
        Adapted from Silsbee & Rafikov (2015) with added damping to fixed point
        """
        if e_disk_eff is None:
            e_disk_eff = -self.B_d(p)/self.A_d(p)
        if pomega_disk is None:
            pomega_disk = self.pomega
        if prec_A is None:
            prec_A = self.A_d(p)
            
        k = p.e*np.cos(p.pomega)
        h = p.e*np.sin(p.pomega)
        k_diff = k - e_disk_eff*np.cos(pomega_disk)
        h_diff = h - e_disk_eff*np.sin(pomega_disk)
        k_dot = k_diff/self.tau_e(p) - h_diff*prec_A
        h_dot = h_diff/self.tau_e(p) + k_diff*prec_A
        e_dot = (k*k_dot + h*h_dot)/np.sqrt(k**2 + h**2)
        pomega_dot = 1/(1 + (h/k)**2)*(h_dot/k - h/k**2*k_dot)
        return e_dot, 2*np.pi/pomega_dot

    def apply_forces(self, p: rebound.Particle, scaling_factor: float) -> None:
        """
        Apply the disk forces to a particle
        """
        p.params['tau_a'] = self.tau_a(p)
        p.params['e_dot'], p.params['tau_omega'] = self.disk_e_damping(p)
        p.params['tau_a'] *= scaling_factor
        p.params['e_dot'] /= scaling_factor
        p.params['tau_omega'] *= scaling_factor

class BasicEccentricDisk():
    """Class to compute planet-disk interactions in a highly simplified eccentric disk"""
    def __init__(self, e0: float, q: float, fixed_tau_e: float, fixed_tau_a: float, 
                Mstar=1., pomega=0., aout=1., G=39.42):
        """
        Parameters
        --------
        e0 : float
            Eccentricity at reference radius
        q : float
            Power law index of eccentricity
        Mstar : float
            Mass of the star, default 1Msun
        pomega : float
            Disk longitude of pericenter, assumed to be constant with radius
        aout : float
            Reference radius of disk
        G : float
            Gravitational constant, default is in AU^3 yr^-2 Msun^-1
        fixed_tau_e, fixed_tau_a : float
            Hard coded eccentricty and semi-major axis timescales
        """
        self.aout = aout
        self.e0, self.q, self.pomega = e0, q, pomega
        self.fixed_tau_a, self.fixed_tau_e = fixed_tau_a, fixed_tau_e
        self.G, self.Mstar = G, Mstar

    def e_d(self, r: float) -> float:
        """Get the disk eccentricity at r"""
        if r < 0:
            return np.nan
        return self.e0*(self.aout/r)**self.q

    def tau_a(self, p: rebound.Particle):
        return self.fixed_tau_a if p.index == 2 else np.inf

    def tau_e(self, p: rebound.Particle):
        return self.fixed_tau_e

    def disk_e_damping(self, p: rebound.Particle) -> tuple[float,float]:
        """
        Compute eccentricity derivative and precession timescale for a particle in the disk
        Adapted from Silsbee & Rafikov (2015) with added damping to fixed point
        """ 
        k = p.e*np.cos(p.pomega)
        h = p.e*np.sin(p.pomega)
        k_diff = k - self.e_d(p.d)*np.cos(self.pomega)
        h_diff = h - self.e_d(p.d)*np.sin(self.pomega)
        k_dot = k_diff/self.tau_e(p)
        h_dot = h_diff/self.tau_e(p)
        e_dot = (k*k_dot + h*h_dot)/np.sqrt(k**2 + h**2)
        pomega_dot = 1/(1 + (h/k)**2)*(h_dot/k - h/k**2*k_dot)
        tau_pomega = 2*np.pi/pomega_dot if pomega_dot != 0 else np.inf
        return e_dot, tau_pomega

    def apply_forces(self, p: rebound.Particle, scaling_factor: float) -> None:
        """
        Apply the disk forces to a particle
        """
        p.params['tau_a'] = self.tau_a(p)
        p.params['e_dot'], p.params['tau_omega'] = self.disk_e_damping(p)
        p.params['tau_a'] *= scaling_factor
        p.params['e_dot'] /= scaling_factor
        p.params['tau_omega'] *= scaling_factor

    def describe(self) -> dict:
        return {'e0': self.e0, 'disk_q': self.q, 'pomega': self.pomega, 'fixed_tau_a': self.fixed_tau_a, 'fixed_tau_e': self.fixed_tau_e}

class BasicCircularDisk():
    """Class to compute planet-disk interactions in a highly simplified disk"""
    def __init__(self, fixed_tau_e: float, fixed_tau_a: float):
        """
        Parameters
        --------
        fixed_tau_e, fixed_tau_a : float
            Hard coded eccentricty and semi-major axis timescales
        """
        self.fixed_tau_a, self.fixed_tau_e = fixed_tau_a, fixed_tau_e

    def tau_a(self, p: rebound.Particle):
        return self.fixed_tau_a if p.index == 2 else np.inf

    def tau_e(self, p: rebound.Particle):
        return self.fixed_tau_e

    def apply_forces(self, p: rebound.Particle, scaling_factor: float) -> None:
        """
        Apply the disk forces to a particle
        """
        p.params['tau_a'] = self.tau_a(p)
        p.params['tau_e'] = self.tau_e(p)
        p.params['tau_a'] *= scaling_factor
        p.params['tau_e'] *= scaling_factor

    def describe(self) -> dict:
        return {'fixed_tau_a': self.fixed_tau_a, 'fixed_tau_e': self.fixed_tau_e}

class StochasticDisk():
    """Class to compute planet-disk interactions in a disk with stochastic forces"""
    def __init__(self, fixed_tau_a: float, fixed_tau_e: float, kappa: float) -> None:
        """
        Parameters
        --------
        fixed_tau_e, fixed_tau_a : float
            Hard coded eccentricty and semi-major axis timescales
        kappa : float
            Strength of stochasticity, in units of stellar gravity
        """
        self.fixed_tau_a, self.fixed_tau_e = fixed_tau_a, fixed_tau_e
        self.kappa = kappa

    def tau_a(self, p: rebound.Particle):
        # no migration for inner planet
        return self.fixed_tau_a if p.index == 2 else np.inf

    def tau_e(self, p: rebound.Particle):
        return self.fixed_tau_e

    def apply_forces(self, p: rebound.Particle, scaling_factor: float) -> None:
        """
        Apply the disk forces to a particle
        """
        p.params['tau_a'] = self.tau_a(p)*scaling_factor
        p.params['tau_e'] = self.tau_e(p)*scaling_factor
        p.params['kappa'] = self.kappa/scaling_factor

    def describe(self) -> dict:
        return {'fixed_tau_a': self.fixed_tau_a, 'fixed_tau_e': self.fixed_tau_e, 'kappa': self.kappa}

class PlanetesimalDisk():
    """Class to represent a disk of planetesimals"""
    def __init__(self, N_pts, disk_mass, loc='outside', Sigma_slope=-2., dyn_T=0.01):
        """
        Parameters
        --------
        N_pts : int
            Number of planetesimals in the disk
        disk_mass : float
            Mass of the disk in sim units
        loc : str
            Location of the planetesimals, either 'inside', 'outside', or 'around'
        Sigma_slope : float
            Power law slope of the surface density profile
        """
        self.N_pts = N_pts
        self.disk_mass = disk_mass
        self.loc = loc
        self.planetesimals_added = False
        self.Sigma_slope = Sigma_slope
        self.dyn_T = dyn_T
    def add_planetesimals(self, sim):
        if self.loc == 'around':
            r_in = sim.particles[1].a*(1/3)**(2/3)
            r_out = sim.particles[2].a*(3/1)**(2/3)
        elif self.loc == 'outside':
            r_in = sim.particles[2].a + 3*sim.particles[2].rhill
            r_out = 2*r_in
        elif self.loc == 'inside':
            r_out = sim.particles[1].a - 3*sim.particles[1].rhill
            r_in = 0.5*r_out
        else:
            raise ValueError('loc must be "around", "outside", or "inside"')
        for i in range(self.N_pts):
            sim.add(m=self.disk_mass/self.N_pts, a=self.draw_radius(r_in, r_out), inc=np.random.rand()*self.dyn_T, 
                    e=np.random.rand()*self.dyn_T, Omega='uniform', l='uniform', omega='uniform')
        self.planetesimals_added = True

    def draw_radius(self, r_in, r_out):
        """Draw a radius from a power-law distribution"""
        if self.Sigma_slope == -2:
            return r_in*(r_out/r_in)**np.random.rand()
        else:
            return (r_in**(self.Sigma_slope+2) + (r_out**(self.Sigma_slope+2) - r_in**(self.Sigma_slope+2))*np.random.rand())**(1/(self.Sigma_slope+2))

    def save_orbits(self, sim):
        """Save the planetesimal orbits"""
        self.orbits = pd.DataFrame([[p.a, p.e, p.inc, p.Omega, p.omega, p.f] for p in sim.particles[sim.N_active:]], columns=['a', 'e', 'i', 'Omega', 'omega', 'f'])

    def describe(self):
        return {'N_pts': self.N_pts, 'pl_disk_mass': self.disk_mass, 'loc': self.loc, 'Sigma_slope': self.Sigma_slope}

def analytical_path(times, e0, pomega0, e_disk, pomega_disk, A, tau_e):
    initial_e_diff = np.sqrt((e0*np.cos(pomega0) - e_disk*np.cos(pomega_disk))**2 + (e0*np.sin(pomega0) - e_disk*np.sin(pomega_disk))**2)
    initial_phi = np.arctan2(e0*np.sin(pomega0) - e_disk*np.sin(pomega_disk), e0*np.cos(pomega0) - e_disk*np.cos(pomega_disk))
    analytical_k = e_disk*np.cos(pomega_disk) + initial_e_diff*np.exp(times/tau_e)*np.cos(A*times+initial_phi)
    analytical_h = e_disk*np.sin(pomega_disk) + initial_e_diff*np.exp(times/tau_e)*np.sin(A*times+initial_phi)
    return analytical_k, analytical_h