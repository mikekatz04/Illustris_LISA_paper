import numpy as np 
from astropy.cosmology import Planck15 as cosmo

class MergerRate:

	def __init__(self, z_vals, Vc, dz=0.001, zmax=10.0):

		self.Vc = Vc
		self.z_vals = z_vals
		self.dz = dz

		self.zs_i = np.arange(0.0, 10.0+dz, dz)
		self.zs_i_plus_1 = self.zs_i + dz
		self.zs = self.zs_i + dz/2.

		####### UNITS UNITS UNITS ?!?!?!?!?!?!?!? ########

		self.dVc_dz = (cosmo.comoving_volume(self.zs_i_plus_1).value - cosmo.comoving_volume(self.zs_i).value)/dz
		self.dz_dt = dz/(cosmo.age(self.zs_i_plus_1).value*1e9 - cosmo.age(self.zs_i).value*1e9)
		self.factor = self.dz_dt*self.dVc_dz/(1+self.zs)

	def weights(self):
		factors = np.abs(np.interp(self.z_vals, self.zs, self.factor))
		return factors/factors.sum()

	def merger_rate(self):
		trans = np.searchsorted(self.zs, self.z_vals)
		uni_vals, uni_num = np.unique(trans, return_counts=True)

		self.dN_dzdVc = np.zeros_like(self.zs)

		self.dN_dzdVc[uni_vals] = uni_num/(self.dz*self.Vc)

		integrand = self.dN_dzdVc*self.factor
		self.rate = np.trapz(integrand[::-1], x=self.zs[::-1])
		return self.rate

