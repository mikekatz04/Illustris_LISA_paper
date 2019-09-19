from scipy.interpolate import interp1d, interp2d
from scipy.special import gamma as Gamma_Function
from scipy.integrate import quad
from scipy.special import hyp2f1
from scipy import constants as ct
import numpy as np
import pdb

from utils.mbhbinaries import mass_ratio_func, MassiveBlackHoleBinaries, AnalyticApproximations

Msun = 1.989e30


class EvolveFDFA(MassiveBlackHoleBinaries):
        # put all quantities into arrays to determine which is major galaxy and minor galaxy'

    def __init__(self, fname, e_0=0.0):

        data = np.genfromtxt(fname, names=True, dtype=None)

        evolve_key_dict = {'m1': 'mass_new_prev_in',
                           'm2': 'mass_new_prev_out',
                           'z': 'redshift',
                           'separation': 'separation',
                           'star_gamma': 'star_gamma',
                           'vel_disp_1': 'vel_disp_prev_in',
                           'vel_disp_2': 'vel_disp_prev_out'}

        for key, col_name in evolve_key_dict.items():
            setattr(self, key, data[col_name])

        # find index of major and minor
        major_1 = (self.m1 >= self.m2)
        major_2 = (self.m1 < self.m2)

        # major black hole mass
        self.M = self.m1*major_1 + self.m2*major_2
        # minor black hole mass
        self.m = self.m1*major_2 + self.m2*major_1

        self.m1 = self.M
        self.m2 = self.m

        # small s denotes secondary,small m is primary (same as paper)
        self.vel_disp_m = self.vel_disp_1*major_1 + self.vel_disp_2*major_2
        self.vel_disp_s = self.vel_disp_1*major_2 + self.vel_disp_2*major_1

        # find large scale orbital decay time
        self.gamma = np.clip(self.star_gamma, 0.55, 2.49)

        # r_infl determined analytically
        self.r_infl = AnalyticApproximations.influence_radius(self.M, self.vel_disp_m)

        self.R_e_m = self.separation

        self.q = mass_ratio_func(self.M, self.m)

        self.e_0 = e_0

        self.Lambda = (np.clip(2**(3./2.)*(self.vel_disp_m/self.vel_disp_s),
                       np.exp(2.0), np.exp(6.0)))

        self.b = self.gamma - 3./2.

    def calculate_timescale(self, return_arr=False):
        self.FD_FA_large_scale_orbital_decay_timescale()
        self.FD_FA_Dynamical_Friction_timescale()
        self.FD_FA_hardening_timescale()

        # TODO: add eccentricity capabilities
        self.e_f = self.e_0

        if return_arr:
            return (np.asarray([self.large_scale_decay_time,
                    self.DF_timescale, self.Hardening_timescale]).T)

        return (self.large_scale_decay_time + self.DF_timescale +
                self.Hardening_GW_timescale, self.e_f)

    def T_gx_star_1(self):
        return (1e9*(0.06 * (2./np.log(self.Lambda)) * (self.R_e_m/10.0)**2 * (self.vel_disp_m/300.)
                * (1e8/self.m)))  # yr

    def T_gx_star_2(self):
        return (1e9*(0.15 * (2./np.log(self.Lambda)) * (self.R_e_m/10.0) *
                (self.vel_disp_m/300.)**2 * (100./self.vel_disp_s)**3))  # yr

    def FD_FA_large_scale_orbital_decay_timescale(self):
        # Equations 54, 56, 57
        out = np.asarray([self.T_gx_star_1(), self.T_gx_star_2()]).T
        self.large_scale_decay_time = np.max(out, axis=1)
        return

    def alpha_func(self):
        self.alpha = (
            (Gamma_Function(self.gamma + 1)/Gamma_Function(self.gamma - 1/2)) *
            (4./3.) * np.pi**(-1/2) * 2.**(self.b - self.gamma) * self.ksi**3 *
            hyp2f1(3./2., -self.b, 5./2., self.ksi**2/2.))
        return

    def beta_func(self):
        beta_integral_vectorized = np.frompyfunc(beta_integral_func, 2, 1)
        integral = beta_integral_vectorized(self.ksi, self.b).astype(np.float64)

        self.beta = (
            (Gamma_Function(self.gamma + 1)/Gamma_Function(self.gamma - 1/2)) * 4*np.pi**(-1/2) *
            2**-self.gamma * integral)
        return

    def delta_func(self):
        self.delta = ((Gamma_Function(self.gamma + 1)/Gamma_Function(self.gamma - 1/2)) *
                      8*np.pi**(-1/2) * (2**(-self.gamma-1)/(self.b+1))*self.ksi * (0.04**(self.b+1)
                      - (2 - self.ksi**2)**(self.b+1)))
        return

    def T_dot_bare(self):
        # in T_dot_bare the coulomb logarith is set to 6.0
        return (1.5e7 * (6.0*self.alpha + self.beta + self.delta)**-1/((3/2. - self.gamma)
                * (3. - self.gamma)) * (self.chi**(self.gamma - 3./2.) - 1) * (self.M/3e9)**(1/2)
                * (self.m/1e8)**-1 * (self.r_infl/300)**(3/2))

    def T_dot_gx(self):
        return (1.2e7 * (np.log(self.Lambda)*self.alpha + self.beta
                + self.delta)**-1/((3. - self.gamma)**2) * (self.chi**(self.gamma - 3.) - 1)
                * (self.M/3e9) * (100/self.vel_disp_s)**3)  # years

    def find_a_crit(self):
        return self.r_infl*(self.m/(2*self.M))**(1/(3 - self.gamma))  # pc

    def FD_FA_Dynamical_Friction_timescale(self, use_interp=False):
        # a_crit = find_a_crit(r_infl, m, M, gamma)
        # chi = a_crit/r_infl
        # self.find_a_h()
        # self.chi = self.a_h/self.r_infl

        self.find_chi()

        self.ksi = 1.
        self.alpha_func()
        self.beta_func()
        self.delta_func()

        out = np.asarray([self.T_dot_bare(), self.T_dot_gx()]).T

        self.DF_timescale = np.min(out, axis=1)
        return

    """
    # Eccentricity related functions
    def k_func(M, m):
        return 0.6 + 0.1*np.log10((M + m) / 3e9)

    def p_e_func(e, M, m):
        k = k_func(M,m)
        return (1-e**2)*(k + (1-k) * (1-e**2)**4)
    """

    def FD_FA_hardening_timescale(self, psi=0.3, phi=0.4):
        #  Equations 61-63
        # includes gravitational regime

        # Below timescale is from FD's paper
        # if e ==0:
        #    p_e = 1.0
        # else:
            # p_e = p_e_func(e, M, m)
        # T_h_GW = (1.2e9 * (r_infl/300.)**((10 + 4*psi)/(5 + psi))
        #   * ((M+m)/3e9)**((-5-3*psi)/(5+psi)) * phi**(-4/(5+psi))
        #   * (4*q/(1+q)**2)**((3*psi - 1)/(5 + psi))* p_e) #years

        # We decided to use Vasiliev's eq. 25

        # f_e = self.f_e_func() * (self.e_f !=0.0) + 1.0 * (self.e_f==0.0)
        f_e = 1.0

        T_h_GW = (
            1.7e8 * (self.r_infl/30.)**((10 + 4*psi)/(5 + psi))
            * ((self.M+self.m)/1e8)**((-5-3*psi)/(5+psi)) * phi**(-4/(5+psi))
            * (4*self.q/(1+self.q)**2)**((3*psi - 1)/(5 + psi))
            * f_e**((1+psi)/(5+psi)) * 20**psi)  # years

        self.T_GW = self.find_T_GW()
        self.Hardening_GW_timescale = T_h_GW*(self.q >= 1e-3) + self.T_GW*(self.q < 1e-3)
        return

    def find_chi(self):
        # combines equation 3b from Merritt et al 2009 and hardening timescale from merritt 2013
        self.chi = 0.25*(self.q/(1.+self.q))
        return

    def find_a_h(self):
        self.a_h = (36.0 * (self.q/(1+self.q)**2) * ((self.M + self.m)/3e9)
                    * (self.vel_disp_m/300)**-2)  # pc
        return

    """
    # Eccentricity related
    def f_e_func(self):
        return (1-self.e_f**2)**(7/2)/(1 + (73./24.)*self.e_f**2 + (37.0/96.)*self.e_f**4)
    """

    def find_T_GW(self):
        self.find_a_h()
        m1 = self.M*Msun*ct.G/ct.c**2
        m2 = self.m*Msun*ct.G/ct.c**2
        beta = 64./5. * m1*m2*(m1+m2)
        T_GW_meters = self.a_h**4/beta
        T_GW_seconds = T_GW_meters/ct.c
        T_GW_yr = T_GW_seconds/(ct.Julian_year)
        return T_GW_yr


def beta_integrand_func(x, ksi, b):
    return x**2 * (2-x**2)**b * np.log((x + ksi)/(x - ksi))


def beta_integral_func(ksi, b):
    integral, err = quad(beta_integrand_func, ksi, 1.4, args=(ksi, b))
    return integral
