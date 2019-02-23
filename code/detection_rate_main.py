import pdb
from collections import OrderedDict
import time

import numpy as np
from astropy.cosmology import Planck15 as cosmo
import scipy.interpolate as interpolate

from utils.mergerrate import MergerRate
from utils.mbhbinaries import MassiveBlackHoleBinaries, AnalyticApproximations, mass_ratio_func
from utils.evolveFDFA import EvolveFDFA
from utils.basicmergers import MagicMergers
from utils.resample import KDEResample, GenerateCatalog

from gwsnrcalc.gw_snr_calculator import snr


def z_at(coalescence_time, num_interp_points=1000):
    # get z_coal
    zs = np.linspace(0.0, 20.0, num_interp_points)
    age = cosmo.age(zs).value*1e9

    check = interpolate.interp1d(age, zs)
    return check(coalescence_time)


def detection_rate_main(
                        num_catalogs, t_obs, duration, fp, evolve_key_guide, kde_key_guide,
                        evolve_class, merger_rate_kwargs, snr_kwargs,
                        only_detectable=True, snr_threshold=8.0):

    begin_time = time.time()

    input_data = np.genfromtxt(fp, names=True, dtype=None)

    evolve_dict = {key: input_data[evolve_key_guide[key]] for key in evolve_key_guide.keys()}

    mbh = evolve_class(**evolve_dict)
    mbh.evolve()

    # mergers in hubble time
    inds_keep = np.where(mbh.coalescence_time < cosmo.age(0.0).value*1e9)[0]

    # merger_rate_kwargs['z_vals'] = mbh.z_coal =
    # np.interp(mbh.coalescence_time[inds_keep], age, zs)
    merger_rate_kwargs['z_vals'] = mbh.z_coal = z_at(mbh.coalescence_time[inds_keep])

    # Merger Rate Per Year ####
    mr_class = MergerRate(**merger_rate_kwargs)
    merger_rate = mr_class.merger_rate()
    print('merger rate:', merger_rate)

    # Prepare KDE

    input_to_kde = np.asarray(
                              [input_data[kde_key_guide['m1']][inds_keep],
                               input_data[kde_key_guide['m2']][inds_keep], mbh.z_coal]).T
    kde_weights = mr_class.weights()

    # kde_kwargs = {'names':kde_key_guide.keys(), 'data': input_to_kde, 'weights':kde_weights}

    kde = KDEResample(data=input_to_kde, weights=kde_weights, names=['m1', 'm2', 'z_coal'])
    kde.make_kernel(bound=1e-6)

    # Generate Catalog
    gc = GenerateCatalog(poisson_parameter=merger_rate, duration=duration, binary_kde=kde)
    gc.make_catalogs(num_catalogs=num_catalogs)

    # Find SNRs

    # TODO: REMOVING HIGH MASS RATIOS BECAUSE PHENOMD NOT SUITABLE. CHECK THESE
    inds_keep = np.where(mass_ratio_func(gc.m1, gc.m2) > 1e-4)[0]

    for name in ['catalog_num', 't_event', 'm1', 'm2', 'z_coal']:
        setattr(gc, name, getattr(gc, name)[inds_keep])

    # start and end time of waveform
    st = gc.t_event
    et = 0.0*((st - t_obs) < 0.0) + (st - t_obs)*((st - t_obs) >= 0.0)

    spin = snr_kwargs['spin']
    snr_out = snr(gc.m1, gc.m2, spin, spin, gc.z_coal, st, et, **snr_kwargs)

    names = 'cat,t_event,m1,m2,z_coal,snr,snr_ins,snr_mr'

    if only_detectable:
        inds_keep = np.where(snr_out[snr_kwargs['sensitivity_curve'] + '_wd_all'] > 8.0)[0]

    else:
        inds_keep = np.arange(len(snr_out[snr_kwargs['sensitivity_curve'] + '_wd_all']))

    output = np.core.records.fromarrays(
                [gc.catalog_num[inds_keep], gc.t_event[inds_keep],
                 gc.m1[inds_keep], gc.m2[inds_keep], gc.z_coal[inds_keep],
                 snr_out[snr_kwargs['sensitivity_curve'] + '_wd_all'][inds_keep],
                 snr_out[snr_kwargs['sensitivity_curve'] + '_wd_ins'][inds_keep],
                 (snr_out[snr_kwargs['sensitivity_curve'] + '_wd_mrg'][inds_keep]**2
                  + snr_out[snr_kwargs['sensitivity_curve'] + '_wd_rd'][inds_keep]**2)**(1/2)],
                names=names)

    print('Total Duration:', time.time()-begin_time)
    return output


if __name__ == "__main__":

    num_catalogs = 10
    t_obs = 5.0  # years
    duration = 100.0  # years
    fp = 'simulation_input_data.txt'

    kde_key_guide = OrderedDict()
    kde_key_guide['m1'] = 'mass_new_prev_in'
    kde_key_guide['m2'] = 'mass_new_prev_out'
    # kde_key_guide['z'] = 'redshift'

    evolve_key_guide = {
                        'm1': 'mass_new_prev_in', 'm2': 'mass_new_prev_out', 'z': 'redshift',
                        'separation': 'separation', 'star_gamma': 'star_gamma',
                        'vel_disp_1': 'vel_disp_prev_in',  'vel_disp_2': 'vel_disp_prev_out'}

    evolve_class = EvolveFDFA

    # evolve_key_guide = {'z':'redshift'}
    # evolve_class = MagicMergers

    merger_rate_kwargs = {'Vc': 106.5**3, 'dz': 0.001, 'zmax': 10.0}

    snr_kwargs = {'spin': 0.8, 'signal_type': ['all', 'ins', 'mrg', 'rd'],
                  'wd_noise': 'HB_wd_noise', 'num_points': 2048, 'add_wd_noise': 'True',
                  'sensitivity_curves': 'LPA', 'prefactor': np.sqrt(16./5.),
                  'num_processors': -1, 'num_splits': 1000, 'verbose': 20}

    check = detection_rate_main(
                num_catalogs, t_obs, duration, fp, evolve_key_guide, kde_key_guide, evolve_class,
                merger_rate_kwargs, snr_kwargs, only_detectable=False,
                snr_threshold=8.0)

    #import matplotlib.pyplot as plt

    #plt.hist()

    import pdb
    pdb.set_trace()
