import scipy.constants as ct
import numpy as np
import pdb

from astropy.cosmology import FlatLambdaCDM

cosmo = FlatLambdaCDM(H0=70.4, Om0=0.2726, Ob0=0.0456)


def mass_ratio_func(m1, m2):
    up = (m1 >= m2)
    down = (m1 < m2)
    return up * (m2/m1) + down * (m1/m2)


class MassiveBlackHoleBinaries:

    def evolve(self):
        self.formation_time = cosmo.age(self.z).value*1e9
        self.t_delay, self.e_f = self.calculate_timescale()
        self.coalescence_time = self.formation_time + self.t_delay
        return

    def readout(self, keys, fp_out):
        out = np.array([getattr(self, key) for key in keys]).T
        out_key = ''
        for key in keys:
            out_key += key + '\t'

        np.savetxt(fp_out, out, header=out_key)
        return


class AnalyticApproximations:

    def half_radius(M_p, z):
        # from FD
        # factor_1=1.0
        factor_1 = 3.0/(3.**(0.73))
        M_p_0 = M_p/((1.+z)**(-0.6))
        factor_2 = (M_p_0/(1.E+8))**0.66

        half_radius = factor_1*factor_2*(1.+z)**(-0.71)

        return half_radius

    def velocity_dispersion(M_p, z):

        factor_1 = 190.0

        M_p_0 = M_p/((1.+z)**(-0.6))
        # velocity_dispersion=factor_1*factor_2*(1.+z)**(0.44)

        # velocity_dispersion=(M_p/1.E+8/1.66)**(1./5.)*200.  used before

        factor_2 = (M_p_0/(1.E+8))**0.2

        velocity_dispersion = factor_1*factor_2*(1.+z)**(0.056)

        return velocity_dispersion

    def influence_radius(M_p, vel_disp):

        # inf_radius=G*M_p/(velocity_dispersion(M_p,z)**2)
        # inf_radius=35.*(M_p/1.E+8)**(0.56)
        # inf_radius=13.*(M_p/1.e8)/(velocity_dispersion(M_p,z)/200.)**2
        inf_radius = 10.8*(M_p/1e8)*(vel_disp/200.0)**-2  # Eq 3b in Merritt et al 2009
        return inf_radius
