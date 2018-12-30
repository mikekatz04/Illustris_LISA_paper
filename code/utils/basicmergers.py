import numpy as np
from utils.mbhbinaries import MassiveBlackHoleBinaries


class MagicMergers(MassiveBlackHoleBinaries):
    def __init__(self, m1, m2, z):
        self.z = z
        self.m1 = m1
        self.m2 = m2

    def calculate_timescale(self):
        # Magic mergers, no timescale
        return np.full(self.z.shape, 0.0), 0.0


class MagicMergersOrigExtract(MassiveBlackHoleBinaries):
    def __init__(self, m1, m2, z):
        self.z = z
        self.m1 = m1
        self.m2 = m2

    def calculate_timescale(self):
        # Magic mergers, no timescale
        # Mergers with a 10^5 are given long evolution
        timescale = np.zeros_like(self.z)
        fix = (self.m1 < 1e6) | (self.m2 < 1e6)
        return timescale + 1e20*fix, 0.0


class ConstantTime(MassiveBlackHoleBinaries):
    def __init__(self, m1, m2, z):
        self.z = z
        self.m1 = m1
        self.m2 = m2

    def calculate_timescale(self):
        # Constant timescale of 1 Gyr
        return np.full(self.z.shape, 1e9), 0.0
