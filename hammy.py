import numpy as np
import matplotlib.pyplot as plt

G, M = 4*np.pi**2, 1

class ResHam():
    """
    Class to compute Hamiltonian coordinates for a near-resonant system
    """
    def __init__(self, k, m_1, m_2, a_1, a_2, e_1, e_2, pomega_1, pomega_2, lambda_1, lambda_2) -> None:
        self.k = k
        gamma_1, gamma_2 = -pomega_1, -pomega_2
        Lambda_1 = m_1*np.sqrt(G*M*a_1)
        Lambda_2 = m_2*np.sqrt(G*M*a_2)
        a_1_bracket = a_1
        a_2_bracket = (k/(k-1))**(2/3)*a_1_bracket
        Lambda_1_bracket = m_1*np.sqrt(G*M*a_1_bracket)
        Lambda_2_bracket = m_2*np.sqrt(G*M*a_2_bracket)
        Gamma_1 = Lambda_1*(1 - np.sqrt(1 - e_1**2))
        Gamma_2 = Lambda_2*(1 - np.sqrt(1 - e_2**2))
        x_1 = np.sqrt(2*Gamma_1)*np.cos(gamma_1)
        x_2 = np.sqrt(2*Gamma_2)*np.cos(gamma_2)
        y_1 = np.sqrt(2*Gamma_1)*np.sin(gamma_1)
        y_2 = np.sqrt(2*Gamma_2)*np.sin(gamma_2)
        f, g = get_f_g(k, 1)
        alpha = G**2*M*m_1*m_2**3/Lambda_2_bracket**2*f/np.sqrt(Lambda_1_bracket)
        beta = G**2*M*m_1*m_2**3/Lambda_2_bracket**2*g/np.sqrt(Lambda_2_bracket)
        self.u_1 = (alpha*x_1 + beta*x_2)/np.sqrt(alpha**2 + beta**2)
        self.v_1 = (alpha*y_1 + beta*y_2)/np.sqrt(alpha**2 + beta**2)
        self.u_2 = (beta*x_1 - alpha*x_2)/np.sqrt(alpha**2 + beta**2)
        self.v_2 = (beta*y_1 - alpha*y_2)/np.sqrt(alpha**2 + beta**2)
        self.Phi_1 = (self.u_1**2 + self.v_1**2)/2
        self.Phi_2 = (self.u_2**2 + self.v_2**2)/2
        h_1_bracket = 1/(m_1*a_1_bracket**2)
        h_2_bracket = 1/(m_2*a_2_bracket**2)
        eta = ((alpha**2 + beta**2)/(9*(h_1_bracket*(k-1)**2 + h_2_bracket*k**2)**2))**(1/3)
        K = Lambda_1 + (k-1)/k*Lambda_2
        K_tilde = K/eta
        Theta = Lambda_2/k
        self.Omega = Theta - self.Phi_1
        self.Omega_tilde = self.Omega/eta
        self.delta = -(h_2_bracket*k**2*(3 + 2*self.Omega_tilde) + h_1_bracket*(k-1)*(3*k - 2*K_tilde + 2*(k-1)*self.Omega_tilde - 3))*(3*(h_1_bracket*(k-1)**2 + h_2_bracket*k**2))**-1
        self.Psi_tilde = self.Phi_1/eta
        self.Phi_2_tilde = self.Phi_2/eta
        self.psi = np.arctan2(self.v_1, self.u_1) + self.k*lambda_2 - (self.k-1)*lambda_1
        self.H = 3*(self.delta + 1)*self.Psi_tilde - self.Psi_tilde**2 - 2*np.sqrt(2*self.Psi_tilde)*np.cos(self.psi)
        self.i_to_plot = None

    def get_H_contours(self, index, Psi_list, psi_list):
        # equilibrium points of the Hamiltonian
        p_eq = np.sort(np.roots([-1, 0, 3*(self.delta[index]+1), -2]))
        return 3*(self.delta[index] + 1)*Psi_list - Psi_list**2 - 2*np.sqrt(2*Psi_list)*np.cos(psi_list), p_eq

    @classmethod
    def from_capture_sim(cls, sim):
        """
        Create a ResHam object from a CaptureSimulation object
        """
        ham = cls(sim.p, sim.m1, sim.m2, *sim.orbits[:,:,0].T, *sim.orbits[:,:,1].T, *sim.orbits[:,:,2].T, *sim.orbits[:,:,3].T)
        ham.i_to_plot = np.where(sim.times > sim.disk_end)[0][0]
        return ham
    
    def phase_space_plot(self, plot_origin=True):
        fig, ax = plt.subplots(subplot_kw=dict(projection='polar'), figsize=(10,10))
        if not self.i_to_plot:
            self.i_to_plot = 9*len(self.delta)//10
        psi_list = np.linspace(0, 2*np.pi, 500)
        Psi_max = self.Psi_tilde[self.i_to_plot:].max()*2
        Psi_list = np.linspace(0, Psi_max, 500)
        Psi_list, psi_list = np.meshgrid(Psi_list, psi_list)
        values, p_eq = self.get_H_contours(-1, Psi_list, psi_list)
        ax.contour(psi_list, np.sqrt(2*Psi_list), values, levels=30, zorder=0, colors='k', linewidths=0.5)
        ax.scatter(self.psi[self.i_to_plot:], np.sqrt(2*self.Psi_tilde[self.i_to_plot:]), s=1, c='k')
        if plot_origin:
            ax.scatter(0, 0, s=30, c='b')
        p_eq = p_eq[np.isreal(p_eq)].real
        ax.scatter(np.arctan2(0, p_eq), np.abs(p_eq), s=10, c='r', marker='x')
        if len(p_eq) == 3:
            # plot the separatrix if there are 3 roots
            sep, _ = self.get_H_contours(-1, p_eq[-1]**2/2, 0)
            ax.contour(psi_list, np.sqrt(2*Psi_list), values, levels=[sep], zorder=0, colors='r', linewidths=3)
        ax.grid(False)
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.set_xlabel(r'$\sqrt{2\tilde{\Psi}}\cos\psi$')
        ax.set_ylabel(r'$\sqrt{2\tilde{\Psi}}\sin\psi$')

def Delta_eq(m1, m2, p, tau_a_rel, tau_e):
    """
    Equilibrium value of Delta in a disk from Terquem & Papaloizou 2019
    Note their definition of Delta differs from ours by a factor of k
    """
    alpha = ((p-1)/p)**(2/3)
    f, g = get_f_g(p, 1)
    A = 3/((p-1)**2*tau_e)*((p-1)/p*m1/(alpha*m2) + 1)*(alpha*m2)**2*(p*f**2+(p-1)**2/p*m1/(alpha*m2)*g**2)
    B = 3/2*(1/tau_a_rel)
    return np.sqrt(-A/B)

def resonance_width(m1, m2, j):
    """
    Resonance with in n2/n1 from Batygin 2015 Eq. 39
    """
    alpha = ((j-1)/j)**(2/3)
    f, g = get_f_g(j, 1)
    return 12*(j/(j-1))*(j*(m1 + m2*alpha**-0.5) - m2*alpha**-0.5)*((m1*g**2 + m2*f**2*(j/(j-1))**(1/3))/(9*(m1*j**2 + m2*(j-1)**(2/3)*j**(4/3))**2))**(1/3)

def approx_delta(k, m_1, m_2, a_1, a_2, e_1, e_2, pomega_1, pomega_2):
    """
    Approximate resonant proximity parameter \delta in the compact orbits approximation Deck & Batygin 2015 Eq. 3
    """
    sigma2 = e_1**2 + e_2**2 - 2*e_1*e_2*np.cos(pomega_1-pomega_2)
    delta_alpha = a_1/a_2 - ((k-1)/k)**(2/3)
    return -1 + 2/3*(sigma2/2 + (delta_alpha)/2/k)*(15*k/(4*(m_1+m_2)))**(2/3)

def eq_Delta(m1, m2, k):
    """
    Maximum value of distance from resonance for separatrix to exist, i.e. delta > 0 at equilibrium value of eccentricity
    """
    peq = -2
    return 3*k**(1/3)/2*(peq**2 - 3)*(4*(m1+m2)/15)**(2/3)

def get_f_g(p: int, q: int) -> tuple[float, float]:
    """Compute the f and g coefficients for the p:p-q resonance"""
    try:
        return {(2,1):(-1.19,0.42),
                (3,1):(-2.03,2.48),
                (4,1):(-2.84,3.28),
                (5,1):(-3.65,4.08),
                (7,2):(-2.81,3.37)}[(p,q)]
    except KeyError:
        return np.nan, np.nan