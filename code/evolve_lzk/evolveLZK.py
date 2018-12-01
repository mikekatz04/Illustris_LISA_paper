"""

ALSO NEED
- mdot
- binned, resolved profiles (not just fits)
- vel-disp fit profiles

"""
# from scipy.interpolate import interp1d, interp2d
# from scipy.special import gamma as Gamma_Function
# from scipy.integrate import quad
# from scipy.special import hyp2f1
import numpy as np
# import pdb

from utils.mbhbinaries import MassiveBlackHoleBinaries
from . import PC, MSOL, radius_schwarzschild


class EvolveLZK(MassiveBlackHoleBinaries):

    NUM_STEPS = 200

    def __init__(self, m1, m2, vel_disp_1, vel_disp_2, star_gamma, separation, redz, e_0=0.0):
        masses = np.array([m1, m2])*MSOL
        self.m1 = masses.max(axis=0)
        self.m2 = masses.min(axis=0)
        self.mtot = self.m1 + self.m2
        self.mrat = self.m2 / self.m1

        self.redz = redz
        self.sep0 = separation*1000*PC
        self.eccen = e_0

        self.num_binaries = self.m1.size

        return

    def init_integral_arrays(self, num_steps):
        sep_extr = [3*radius_schwarzschild(self.m2.min()), self.sep0.max()]
        shape = (self.num_binaries, num_steps)

        self.rads = np.logspace(*np.log10(sep_extr), self.NUM_STEPS)
        self.dadt = np.zeros(shape)
        self.dadt_df = np.zeros(shape)
        self.dadt_sc = np.zeros(shape)
        self.dadt_ct = np.zeros(shape)
        self.dadt_gw = np.zeros(shape)

        return

    def integrate(self, num_steps=None):
        if num_steps is None:
            num_steps = self.NUM_STEPS

        self.init_integral_arrays(num_steps)



        return

    def calculate_timescale(self):
        time_coal = 0.0
        eccen_final = 0.0

        return time_coal, eccen_final
