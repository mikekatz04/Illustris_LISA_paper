import numpy as np
from utils.mbhbinaries import MassiveBlackHoleBinaries


class NoDelay(MassiveBlackHoleBinaries):
    def __init__(self, fname):
        data = np.genfromtxt(fname, names=True, dtype=None)

        evolve_key_dict = {'m1': 'mass_new_prev_in',
                           'm2': 'mass_new_prev_out',
                           'z': 'redshift'}

        for key, col_name in evolve_key_dict.items():
            setattr(self, key, data[col_name])

    def calculate_timescale(self):
        # Magic mergers, no timescale
        return np.full(self.z.shape, 0.0), 0.0


class NoDelayOrigExtract(MassiveBlackHoleBinaries):
    def __init__(self, fname):
        data = np.genfromtxt(fname, names=True, dtype=None)

        evolve_key_dict = {'m1': 'mass_new_prev_in',
                           'm2': 'mass_new_prev_out',
                           'z': 'redshift'}

        for key, col_name in evolve_key_dict.items():
            setattr(self, key, data[col_name])

    def calculate_timescale(self):
        # Magic mergers, no timescale
        # Mergers with a 10^5 are given long evolution
        timescale = np.zeros_like(self.z)
        fix = (self.m1 < 1e6) | (self.m2 < 1e6)
        return timescale + 1e20*fix, 0.0


class ConstantTime(MassiveBlackHoleBinaries):
    def __init__(self, fname):
        data = np.genfromtxt(fname, names=True, dtype=None)

        evolve_key_dict = {'m1': 'mass_new_prev_in',
                           'm2': 'mass_new_prev_out',
                           'z': 'redshift'}

        for key, col_name in evolve_key_dict.items():
            setattr(self, key, data[col_name])

    def calculate_timescale(self):
        # Constant timescale of 1 Gyr
        return np.full(self.z.shape, 1e9), 0.0
