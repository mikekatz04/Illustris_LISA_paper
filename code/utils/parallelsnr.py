from pyphenomd.pyphenomd import snr as snr_calculator
import numpy as np
import multiprocessing as mp
from multiprocessing import Pool
import time

def parallel_snr_func(num_process, snr_args, snr_kwargs, verbose=0):
	if verbose != 0:
		if (num_process + 1) % verbose == 0:
			print('Start process:', num_process)

	snr = snr_calculator(*snr_args, **snr_kwargs)

	if verbose != 0:
		if num_process + 1 % verbose == 0:
			print('End process:', num_process)

	return snr

class ParallelSNR:

	def __init__(self, m1, m2, z, st, et, chi=0.8, chi_1=None, chi_2=None, snr_kwargs={'wd_noise':True, 'num_points':2048, 'sensitivity_curve':'LPA', 'prefactor':np.sqrt(16./5.)}):

		if ((chi_1 == None) & (chi_2 != None)) or ((chi_1 != None) & (chi_2 == None)):
			raise Exception("Either supply `chi`, or supply both `chi_1` and `chi_2`. You supplied only `chi_1` or `chi_2`.")

		if chi_1 == None:
			if type(chi) == float:
				chi = np.full((len(m1),), chi)

			self.chi_1 = chi
			self.chi_2 = chi
		else:
			if type(chi_1) == float:
				chi = np.full((len(m1),), chi_1)
			if type(chi_2) == float:
				chi = np.full((len(m1),), chi_2)

			self.chi_1 = chi_1
			self.chi_2 = chi_2

		self.m1, self.m2, self.z, self.st, self.et = m1, m2, z, st, et
		self.snr_kwargs = snr_kwargs

	def prep_parallel(self, num_processors=None, num_splits=1000, verbose=50):
		if num_processors == None:
			self.num_processors = mp.cpu_count()
		else:
			self.num_processors = num_processors

		self.num_splits = num_splits

		split_val = int(np.ceil(len(self.m1)/num_splits))
		split_inds = [num_splits*i for i in np.arange(1,split_val)]

		
		inds_split_all = np.split(np.arange(len(self.m1)), split_inds)

		self.args = []
		for i, ind_split in enumerate(inds_split_all):
			snr_args = (self.m1[ind_split], self.m2[ind_split], self.chi_1[ind_split], self.chi_2[ind_split], self.z[ind_split], self.st[ind_split], self.et[ind_split])

			self.args.append((i,snr_args, self.snr_kwargs, verbose))

		return

	def run_parallel(self, timer=False):
		if timer:
			start_timer = time.time()

		#for testing
		#check = parallel_snr_func(*self.args[10])
		#import pdb
		#pdb.set_trace()

		print('numprocs', self.num_processors)
		with Pool(self.num_processors) as pool:
			print('start pool with {} processors: {} total processes.\n'.format(self.num_processors, len(self.args)))

			results = [pool.apply_async(parallel_snr_func, arg) for arg in self.args]
			out = [r.get() for r in results]
			out = np.concatenate(out)
		if timer:
			print("SNR calculation time:", time.time()-start_timer)
		return out


