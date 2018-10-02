import numpy as np
from utils.mbhbinaries import MassiveBlackHoleBinaries


class MagicMergers(MassiveBlackHoleBinaries):
	def __init__(self, z):
		self.z = z

	def calculate_timescale(self):
		#Magic mergers, no timescale
		return np.full(self.z.shape, 0.0), 0.0

class ConstantTime(MassiveBlackHoleBinaries):
	def __init__(self, z):
		self.z = z

	def calculate_timescale(self):
		#Magic mergers, no timescale
		return np.full(self.z.shape, 1e9), 0.0