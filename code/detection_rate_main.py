import pdb
from collections import OrderedDict
import time

import numpy as np
from astropy.cosmology import Planck15 as cosmo

from utils.mergerrate import MergerRate
from utils.mbhbinaries import EvolveFDFA, MassiveBlackHoleBinaries, AnalyticApproximations, mass_ratio_func
from utils.resample import KDEResample, GenerateCatalog
from utils.parallelsnr import ParallelSNR, parallel_snr_func

def detection_rate_main(num_catalogs, t_obs, duration, fp, evolve_key_guide, kde_key_guide, evolve_class, merger_rate_kwargs, parallel_kwargs, only_detectable=False, snr_threshold=8.0):

	begin_time = time.time()

	input_data = np.genfromtxt(fp, names=True, dtype=None)

	evolve_dict = {key: input_data[evolve_key_guide[key]] for key in evolve_key_guide.keys()}

	mbh = evolve_class(**evolve_dict)
	mbh.evolve()

	#mergers in hubble time
	inds_keep = np.where(mbh.coalescence_time < cosmo.age(0.0).value*1e9)[0]

	### get z_coal
	zs = np.linspace(0.0, 20.0, 10000)
	age = cosmo.age(zs).value*1e9

	import scipy.interpolate as interpolate

	check = interpolate.interp1d(age, zs)

	#merger_rate_kwargs['z_vals'] = mbh.z_coal = np.interp(mbh.coalescence_time[inds_keep], age, zs)
	merger_rate_kwargs['z_vals'] = mbh.z_coal = check(mbh.coalescence_time[inds_keep])

	#### Merger Rate Per Year ####
	mr_class = MergerRate(**merger_rate_kwargs)
	merger_rate = mr_class.merger_rate()

	#### Prepare KDE ####

	input_to_kde = np.asarray([input_data[kde_key_guide['m1']][inds_keep], input_data[kde_key_guide['m2']][inds_keep], mbh.z_coal]).T
	kde_weights = mr_class.weights()

	#kde_kwargs = {'names':kde_key_guide.keys(), 'data': input_to_kde, 'weights':kde_weights}

	kde = KDEResample(data=input_to_kde, weights=kde_weights, names=['m1', 'm2', 'z_coal'])
	kde.make_kernel(bound=1e-6)

	#### Generate Catalog ####

	#generate_catalog_kwargs = {'poisson_parameter': merger_rate, 'duration':duration, 'binary_kde':kde}
	
	gc = GenerateCatalog(poisson_parameter=merger_rate, duration=duration, binary_kde=kde)
	gc.make_catalogs(num_catalogs=num_catalogs)

	#### Find SNRs ####

	##### REMOVING HIGH MASS RATIOS BECAUSE PHENOMD NOT SUITABLE. CHECK THESE #####
	inds_keep = np.where(mass_ratio_func(gc.m1, gc.m2) > 1e-4)[0]

	for name in ['catalog_num', 't_event', 'm1', 'm2', 'z_coal']:
		setattr(gc, name, getattr(gc, name)[inds_keep])

	#start and end time of waveform
	st = gc.t_event
	et = 0.0*((st - t_obs) < 0.0) + (st - t_obs)*((st - t_obs) >= 0.0)

	para = ParallelSNR(gc.m1, gc.m2, gc.z_coal, st, et, chi=0.8, snr_kwargs={'wd_noise':True, 'num_points':2048, 'sensitivity_curve':'LPA', 'prefactor':np.sqrt(16./5.)})

	para.prep_parallel(**parallel_kwargs)
	snr = para.run_parallel(timer=True)

	names = 'cat,t_event,m1,m2,z_coal,snr'

	if only_detectable:
		inds_keep = np.where(snr>8.0)[0]

	else:
		inds_keep = np.arange(len(snr))

	output = np.core.records.fromarrays([gc.catalog_num[inds_keep], gc.t_event[inds_keep], gc.m1[inds_keep], gc.m2[inds_keep], gc.z_coal[inds_keep], snr[inds_keep]], names=names)

	print('Total Duration:', time.time()-begin_time)
	return output





if __name__ == "__main__":

	num_catalogs = 10000
	t_obs = 10.0 #years
	duration = 100.0 #years
	fp = '../data_ready_june_snap_lim_1.txt'

	kde_key_guide = OrderedDict()
	kde_key_guide['m1'] = 'mass_new_prev_in'
	kde_key_guide['m2'] = 'mass_new_prev_out'
	#kde_key_guide['z'] = 'redshift'

	evolve_key_guide = {'m1':'mass_new_prev_in', 'm2':'mass_new_prev_out', 'z':'redshift', 'separation':'separation', 'gamma':'gamma', 'vel_disp_1':'vel_disp_prev_in',  'vel_disp_2':'vel_disp_prev_out'}
	evolve_class = EvolveFDFA
	merger_rate_kwargs = {'Vc':106.5**3, 'dz':0.001, 'zmax':10.0}

	parallel_kwargs = {'num_processors':None, 'num_splits':1000, 'verbose':10}

	check = detection_rate_main(num_catalogs, t_obs, duration, fp, evolve_key_guide, kde_key_guide, evolve_class, merger_rate_kwargs, parallel_kwargs, only_detectable=False, snr_threshold=8.0)
	import pdb
	pdb.set_trace()

