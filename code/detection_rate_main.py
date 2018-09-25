import pdb
from collections import OrderedDict

import numpy as np
from astropy.cosmology import Planck15 as cosmo

from utils.mergerrate import MergerRate
from utils.mbhbinaries import EvolveFDFA, MassiveBlackHoleBinaries, AnalyticApproximations
from utils.resample import KDEResample, GenerateCatalog



def detection_rate_main(num_catalogs, duration, fp, evolve_key_guide, kde_key_guide, evolve_class, merger_rate_kwargs):

	input_data = np.genfromtxt(fp, names=True, dtype=None)

	evolve_dict = {key: input_data[evolve_key_guide[key]] for key in evolve_key_guide.keys()}

	mbh = evolve_class(**evolve_dict)
	mbh.evolve()

	#mergers in hubble time
	inds_keep = np.where(mbh.coalescence_time < cosmo.age(0.0).value*1e9)[0]

	merger_rate_kwargs['z_vals'] = mbh.z[inds_keep]

	#### Merger Rate Per Year ####
	mr_class = MergerRate(**merger_rate_kwargs)
	merger_rate = mr_class.merger_rate()

	#### Prepare KDE ####

	input_to_kde = np.asarray([input_data[kde_key_guide[key]][inds_keep] for key in kde_key_guide.keys()]).T
	kde_weights = mr_class.weights()

	kde_kwargs = {'names':kde_key_guide.keys(), 'data': input_to_kde, 'weights':kde_weights}

	kde = KDEResample(**kde_kwargs)
	kde.make_kernel(bound=1e-6)

	#### Generate Catalog ####

	generate_catalog_kwargs = {'poisson_parameter': merger_rate, 'duration':duration, 'binary_kde':kde}
	
	gc = GenerateCatalog(**generate_catalog_kwargs)
	cats = gc.make_catalogs(num_catalogs=num_catalogs)






	pdb.set_trace()

	return





if __name__ == "__main__":

	num_catalogs = 100
	duration = 100.0 #years
	fp = '../../simulation/data_ready_june_snap_lim_1.txt'

	kde_key_guide = OrderedDict()
	kde_key_guide['m1'] = 'mass_new_prev_in'
	kde_key_guide['m2'] = 'mass_new_prev_out'
	kde_key_guide['z'] = 'redshift'

	evolve_key_guide = {'m1':'mass_new_prev_in', 'm2':'mass_new_prev_out', 'z':'redshift', 'separation':'separation', 'gamma':'gamma', 'vel_disp_1':'vel_disp_prev_in',  'vel_disp_2':'vel_disp_prev_out'}
	evolve_class = EvolveFDFA
	merger_rate_kwargs = {'Vc':106.5**3, 'dz':0.001, 'zmax':10.0}


	detection_rate_main(num_catalogs, duration, fp, evolve_key_guide, kde_key_guide, evolve_class, merger_rate_kwargs, )