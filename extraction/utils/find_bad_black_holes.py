import numpy as np 
import pdb
import h5py

h=0.704


class FindBadBlackHoles:
	"""
	FindBadBlackHoles walks down trees in the sublink trees locating bad black holes. This is the key aspect to the more in-depth analysis to access the near-seed mass black holes. What we mean by bad black holes is the following: when a subhalo loses its black hole in a fly-by encounter and subsequently spawns a new black hole, the spawned black hole is considered bad. 

		List of the steps we take to do this (this is after we have filtered for black holes that stop existing at snapshots earlier than the last snapshot):

		step 1: start at the subhalo where a black hole is about to cease to exist. Its first descendant will either be the galaxy it has merged into, or it will remain on its own except without a black hole. 

		step 2: iterate down the descendant tree until the descendant index is -1

		(as we iterate down the tree)

		step 3: find if a subhalo we are looking at has a black hole

		step 4: if it has a black hole, check if this black hole's first snapshot (i.e. its formation snapshot) is the same as the snapshot of the current descendant in the tree. 

			- If True: this means the black hole formed into the subhhalo that lost its black hole (making it a bad black hole). 

			- If False: it is not he same as the formation snapshot, that means the subhalo we are looking at merged into another subhalo with a black hole that had existed at previous snapshots. 

		step 5: add bad black holes to list and read out the data

		attributes:
			:param	directory - (str) - directory to work in

		methods:
			get_subs_from_all_bhs
			search_bad_black_holes


		MAIN JOB: locate bad black holes resulting from fly-by encounters of host galaxies
	"""
	def __init__(self, directory='./extraction_files'):
		self.directory = directory

		if 'bad_arr.txt' in os.listdir(self.directory):
			self.needed = False
		else:
			self.needed = True

	def get_subs_from_all_bhs(self):
		"""
		Get information from the all bhs catalog to find the information for the last snapshot black holes exist. This information is also needed as we walk down the sublink tree. 
		"""

		#get initial information and calculate raw subhalo id number according to Illustris convention
		with h5py.File(self.directory + 'bhs_all_new.hdf5', 'r') as f_all_bhs:
			part_ids_all = f_all_bhs['ParticleIDs_new'][:]
			snaps_all = f_all_bhs['Snapshot'][:]
			subhalos_all = f_all_bhs['Subhalo'][:]
			subID_raw_all = np.asarray(snaps_all*1e12 + subhalos_all, dtype=np.int64)
			bh_masses_all = f_all_bhs['BH_Mass'][:]

		#find the last time you find a unique id
		ids_unique, index = np.unique(part_ids_all[::-1], return_index=True)

		#only need to look at black holes where the last time
		# we see them are before snapshot 135
		filter_inds = np.where(snaps_all[::-1][index] != 135)[0]

		#this is the filtered version of subID_raw_all
		subID_raw_filtered = np.asarray(snaps_all[::-1][index][filter_inds]*1e12 + subhalos_all[::-1][index][filter_inds], dtype=np.int64)

		return subID_raw_filtered, part_ids_all, snaps_all, subID_raw_all, bh_masses_all

	def search_bad_black_holes(self):
		"""
		Find bad black holes using the method described in the FindBadBlackHoles class description. 
		"""

		#get all the information from the all black holes catalog needed
		subID_raw_filtered, part_ids_all, snaps_all, subID_raw_all,bh_masses_all = self.get_subs_from_all_bhs()

		#open the sublink tree file. Keep it open because it is 
		#too much information to store it all in memory
		f_sublink = h5py.File(self.directory + 'sublink_short.hdf5', 'r')

		#get the raw IDs from sublink and sort them
		subID_raw_sublink = np.asarray(f_sublink['SubfindID'][:310600757] + f_sublink['SnapNum'][:310600757]*1e12, dtype=np.int64)

		inds_sort = np.argsort(subID_raw_sublink)

		#find the raw sub IDs for subhalos at the last snapshot the black holes exist.
		#We can do this by index because of our initial work with adding
		#decendant and subhaloID indices in addition to the real values.
		final_subs_inds = inds_sort[np.searchsorted(subID_raw_sublink[inds_sort], subID_raw_filtered)]

		bad = []
		print(len(final_subs_inds), 'to look at')

		for j, final_ind in enumerate(final_subs_inds):
			desc_ind = final_ind

			#iterate down the tree
			while desc_ind != -1:

				#using indices makes this easy (play with these files separately if you uncertain of this procedure)
				#next descendant
				desc_ind = f_sublink['Descendant_index'][desc_ind]

				#check if the descendant will have a black hole in it
				if f_sublink['SubhaloLenType'][desc_ind][5] > 0:
					
					#figure out which subhalo in all bhs catalog corresponds to this sub with a black hole from the sublink catalog
					ind_sub = np.where(subID_raw_all == subID_raw_sublink[desc_ind])[0][0]

					#find all the snapshots that contain the black hole that is in this subhalo of interest
					snaps_bh = snaps_all[np.where(part_ids_all == part_ids_all[ind_sub])]

					#check if the first snapshot this black hole exists is the same as the descendant
					if snaps_bh[0] == f_sublink['SnapNum'][desc_ind]:

						#append to bad list
						bad.append((part_ids_all[ind_sub], bh_masses_all[ind_sub], f_sublink['SnapNum'][desc_ind]))

			if j % 100 == 0:
				print(j)

		# read out
		bad_arr = np.asarray(bad, dtype=[('id', np.dtype(np.uint64)), ('mass', np.dtype(float)), ('snap', np.dtype(np.int32))])

		np.savetxt(self.directory + 'bad_arr.txt', bad_arr)
		f_sublink.close()
		return

