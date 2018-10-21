"""
MAIN PURPOSE: final check for which subhalos we need to download primarily based on resolution requirements.
"""

import os
import numpy as np
import h5py


class FindSubhalosForSearch:
	"""
	FindSubhalosForSearch checks for an outputs the subhalos that need to be downloaded in order to create density profiles and get velocity dispersions. These subhalos have to be downloaded from the Illustris server with all of their particles. This will take a lot of memory, so we only want to do this for the subhalos we absolutely need.

		These are determined based on two criteria:

		1) For each merger, both constituent black holes and the final black hole need to have an associated host. If any of the three do not, we do not consider the merger as part of our catalog. If this is the case, we do not download any of the three subhalos. At this point in this entire extraction process, a vast majority of black holes will pass this criterion because or preprocessing related to requiring black holes to exist at snapshots in subhalos.

		2) We check for resolution. Following Kelley et al 2017 and Blecha et al 2016, we make sure each subhalo we are going to download and include in our study has 300 gas cells, 300 star particles, and 80 dm particles.

		If the mergers/subhalos pass this criteria, the subhalos are added to ``snaps_and_subs_needed.txt``.

		attributes:
			: param	directory - (str) - directory to work in
			: param use_second_sub_back - (bool) - some merger constituent black holes exist in the same galaxy prior to merger. A value of true will look two snapshots back if this is the case. We did not use this in the paper.
			: param skip_snaps - list of (int) - list of bad snapshots (53 and 55 in Illustris-1s)

			needed - (bool) - if this code needs to run

		methods:
			gather_from_merger_file
			gather_from_all_bhs_file
			gather_from_subs_with_bhs
			find_subs_to_search
	"""

	def __init__(self, directory, use_second_sub_back=False, skip_snaps=[53, 55]):
		self.directory = directory
		self.use_second_sub_back = use_second_sub_back
		self.skip_snaps = skip_snaps

		if 'snaps_and_subs_needed.txt' in os.listdir(self.directory):
			self.needed = False

		else:
			self.needed = True

	def gather_from_merger_file(self):
		"""
		Get the merger information from merger file.
		"""

		with h5py.File('bhs_mergers_new.hdf5', 'r') as f_merg:
			id_in_new = f_merg['id_in_new'][:]
			id_out_new = f_merg['id_out_new'][:]
			merg_snap = f_merg['snapshot'][:]

		return id_in_new, id_out_new, merg_snap

	def gather_from_all_bhs_file(self):
		"""
		Gather general black hole information from all bhs catalog.
		"""

		with h5py.File('bhs_all_new.hdf5', 'r') as f_all:
			all_arr = np.core.records.fromarrays([f_all['ParticleIDs_new'][:], f_all['Snapshot'][:], f_all['Subhalo'][:]], names='id, snap, sub')

		# sort the array
		sort_all = np.argsort(all_arr, order=('snap', 'sub'))

		all_arr = all_arr[sort_all]

		return all_arr

	def gather_from_subs_with_bhs(self):
		"""
		Gather information about subhalos with bhs.
		"""

		with h5py.File('subs_with_bhs.hdf5', 'r') as f_gc:
			subID_raw_gc = np.asarray(f_gc['Snapshot'][:]*1e12 + f_gc['SubhaloID'][:], dtype=np.uint64)
			SubhaloLenType = f_gc['SubhaloLenType'][:]

		# sort the array
		sort_gc = np.argsort(subID_raw_gc)

		return subID_raw_gc, SubhaloLenType, sort_gc

	def find_subs_to_search(self):
		"""
		As discussed in the paper, if the bhs do not have an associated halo before or after merger, the merger is not considered.
		"""
		# get all necessary information from files
		id_in_new, id_out_new, merg_snap = self.gather_from_merger_file()
		all_arr = self.gather_from_all_bhs_file()
		subID_raw_gc, SubhaloLenType, sort_gc = self.gather_from_subs_with_bhs

		good = np.genfromtxt('good_mergers.txt').astype(int)

		subs_to_search = []
		print('Search subhalos for', len(good), 'mergers.')
		for i, m in enumerate(good):
			if i % 100 == 0:
				print(i)

			# get the information specific to this merger
			id_in = id_in_new[m]
			id_out = id_out_new[m]
			snap = merg_snap[m]

			# look for host galaxy post merger
			try:
				ind_final = np.where((all_arr['id'] == id_out) & (all_arr['snap'] == snap))[0][0]

			# if not found, just go to next merger because this one will not be considered
			except IndexError:
				continue

			# assign prev_out_sub
			# need to avoid snapshot 53 and 55
			if snap-1 in self.skip_snaps:
				prev_snap = snap-2
			else:
				prev_snap = snap-1

			try:
				ind_prev_out = np.where((all_arr['id'] == id_out) & (all_arr['snap'] == prev_snap))[0][0]

			except IndexError:
				continue

			# prev_in_sub
			try:
				ind_prev_in = np.where((all_arr['id'] == id_in) & (all_arr['snap'] == prev_snap))[0][0]

			except IndexError:
				continue

			# some constituent black holes are in the same subhalo prior to merger. If desired this will go back to the next sub. We do not use this in the paper.
			if self.use_second_sub_back:
				if all_arr['sub'][ind_prev_in] == all_arr['sub'][ind_prev_out]:
					# prev_out_sub
					snap = prev_snap
					if snap-1 in self.skip_snaps:
						prev_snap = snap-2
					else:
						prev_snap = snap-1

					try:
						ind_prev_out = np.where((all_arr['id'] == id_out) & (all_arr['snap'] == prev_snap))[0][0]

					except IndexError:
						continue

					# prev_in_sub
					try:
						ind_prev_in = np.where((all_arr['id'] == id_in) & (all_arr['snap'] == prev_snap))[0][0]

					except IndexError:
						continue

			# if all three subhalos are found, append to subs to search
			subs_to_search.append({'merger': m, 'final_out': ind_final, 'prev_out': ind_prev_out, 'prev_in': ind_prev_in})

		# gather list
		subs_to_search = {key: [subs_to_search[i][key] for i in range(len(subs_to_search))] for key in subs_to_search[0].keys()}

		# convert to arrays
		subs_to_search = {key: np.asarray(subs_to_search[key], dtype=np.dtype(int)) for key in subs_to_search}

		# final_out represents and index to the subhalo in all_arr
		subID_raw_mergers = np.asarray(all_arr['snap'][np.asarray(subs_to_search['final_out'], dtype=int)]*1e12 + all_arr['sub'][np.asarray(subs_to_search['final_out'], dtype=int)], dtype=np.int64)

		# get the index in subs_with_bhs.hdf5 information for all host galaxies post merger
		inds_final_gc = sort_gc[np.searchsorted(subID_raw_gc[sort_gc], subID_raw_mergers)]

		# get number of particles for each type to check resolution
		SubhaloLenType = SubhaloLenType[inds_final_gc]

		# keep only galaxies with specific limits on particle counts
		# 80 gas cells, 80 stars, and 300 dm particles following Kelley et al 2017 and Blecha et al 2016
		keep1 = np.where((SubhaloLenType[:, 0] >= 80) & (SubhaloLenType[:, 1] >= 300) & (SubhaloLenType[:, 4] >= 80))[0]

		# filter out unresolved galaxies
		subs_to_search = {key: subs_to_search[key][keep1] for key in subs_to_search}

		# guide dictionary for output with integers
		which = {'final_out': 3, 'prev_out': 2, 'prev_in': 1}

		# populate an output list to be read out and then concatenate into single array
		out = []
		for key in ['final_out', 'prev_in', 'prev_out']:
			out.append([subs_to_search['merger'], all_arr['snap'][subs_to_search[key]], all_arr['sub'][subs_to_search[key]], np.full(len(subs_to_search[key]), which[key])])

		out = np.concatenate(out, axis=1).T

		# build structured array for sorting
		out = np.core.records.fromarrays([out[:, 0], out[:, 3], out[:, 1], out[:, 2]], dtype=[('m', np.dtype(int)), ('which', np.dtype(int)), ('snap', np.dtype(int)), ('sub', np.dtype(int))])

		out = np.sort(out, order=('m', 'which', 'snap', 'sub'))

		# read out
		np.savetxt('snaps_and_subs_needed.txt', out, fmt='%i\t%i\t%i\t%i', header='which number is (3, final_out) (2, prev_out) (1, prev_in)\nmerger\twhich\tsnap\tsub')

		return
