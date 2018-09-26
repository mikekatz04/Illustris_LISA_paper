import numpy as np
import numpy.random as rm
from scipy.special import logit,expit
from astropy.stats import knuth_bin_width
from sklearn.preprocessing import MinMaxScaler


class KDEResample:
	def __init__(self, data, weights, names=['m1', 'm2', 'z']):
		self.data, self.weights, self.names =  data, weights, names
		
	def make_kernel(self, bound=1e-6):
		#kde_names = self.sid['needed_values']

		bound = bound
		self.scale_data(0.0+bound, 1.0-bound)		
		self.data = logit(self.data)
		self.knuth_bandwidth_determination()
		self.covariance = np.cov(self.data.T, aweights=self.weights)*self.bw**2
		#getattr(self, self.sid['mc_generation_info']['kde_method'])()
		#self.kernel = KDEReturn(self.kernel_class, self.scaler, kde_names)
		#self.mc_kernel = KDEReturn(self.kernel_class, self.scaler, kde_names)
		return

	def knuth_bandwidth_determination(self, bw_selection='min'):
		#bandwidth selection is min, max, mean
		bandwidths = np.asarray([knuth_bin_width(data_set) for data_set in self.data.T])
		self.bw = getattr(bandwidths, bw_selection)()
		return

	def scale_data(self, min_val=0.0, max_val=1.0):
		self.scaler = MinMaxScaler(feature_range=(min_val, max_val), copy=True)
		self.data = self.scaler.fit_transform(self.data)
		return

	def resample(self, size=1):

		norm = rm.multivariate_normal(np.zeros((self.data.T.shape[0],), float), self.covariance, size=size)
		indices = rm.choice(np.arange(len(self.data)), size=size, replace=True, p=self.weights/self.weights.sum())

		means = self.data[indices]

		data_drawn = means + norm
		data_drawn = expit(data_drawn)
		data_drawn = self.scaler.inverse_transform(data_drawn)

		#correct for scikit vs scipy kde
		return data_drawn



class GenerateCatalog:

	def __init__(self, poisson_parameter, duration, binary_kde):
		self.poisson_parameter, self.duration, self.binary_kde = poisson_parameter, duration, binary_kde

	def make_catalogs(self, num_catalogs=1):
		poisson_par = self.poisson_parameter*self.duration
		self.num_events = rm.poisson(lam=poisson_par, size=num_catalogs)

		#distribute uniformly over time
		self.t_event = rm.uniform(low=0.0, high=self.duration, size=self.num_events.sum())
		binary_parameters_event = self.binary_kde.resample(size=self.num_events.sum())
		for i, name in enumerate(self.binary_kde.names):
			setattr(self, name, binary_parameters_event[:,i])

		self.catalog_num = np.repeat(np.arange(num_catalogs), self.num_events)
		return

if __name__ == "__main__":
	from astropy.cosmology import Planck15 as cosmo
	import pdb
	import time
	import matplotlib.pyplot as plt 

	names = ['z', 'm1', 'm2']
	data = np.genfromtxt('../../../simulation/data_ready_june_snap_lim_1.txt', names=True, dtype=None)
	z_coal = data['redshift']

	dz = 0.001
	zs_i = np.arange(0.0, 8.0+dz, dz)
	zs_i_plus_1 = zs_i + dz
	zs = zs_i + dz/2.

	####### UNITS UNITS UNITS ########

	dVc_dz = (cosmo.comoving_volume(zs_i_plus_1).value - cosmo.comoving_volume(zs_i).value)/dz
	dz_dt = dz/(cosmo.age(zs_i_plus_1).value*1e9 - cosmo.age(zs_i).value*1e9)

	weights = np.abs(np.interp(z_coal, zs, dz_dt*dVc_dz/(1+zs)))
	kde = KDEResample(np.asarray([data['redshift'], data['mass_new_prev_in'], data['mass_new_prev_out']]).T, weights, names)
	kde.make_kernel()

	st = time.time()
	gc = GenerateCatalog(0.8, 100.0, kde, num_catalogs=100000)
	cats = gc.make_catalogs()
	print(time.time() -st)

	plt.hist(np.log10(data['mass_new_prev_in']), bins=30, weights=weights, density=True, histtype='step', log=True)
	plt.hist(np.log10(cats['m1']), bins=30, density=True, histtype='step', log=True)
	plt.show()
	pdb.set_trace()







