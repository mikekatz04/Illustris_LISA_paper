"""
MAIN PURPOSE: gather subhalos with black holes and create ``subs_with_bhs.hdf5`` file.
"""
import h5py
import os
import numpy as np

from utils.generalfuncs import get


class GetGroupSubs:
	"""
	GetGroupSubs downloads the necessary group catalog files from the Illustris server. It gathers the information we are interested in and outputs to a file. It then deletes that snapshots group catalog file. This code is designed to pick up where it left off if the downloads time out.

		attributes:
			:param	first_snap_with_bhs - (int) - the first snapshot where black holes appear in the simulation
			:param	additional_keys - list of (str) - keys of interest, not including Snapshot, SubhaloID, or SubhaloLenType
			:param	directory - (str) - directory to store final files in
			:param	ill_run - (int) - integer representing illustris run
			:param	snaps_to_skip - list of (int) - snapshots that are missing (53, 55 in Illustris-1)

			needed - (bool) - does this code needed to run


		methods:
			download_and_add_file_info
	"""

	def __init__(self, first_snap_with_bhs=30, snaps_to_skip=[53, 55], additional_keys=['SubhaloCM', 'SubhaloMassType', 'SubhaloPos', 'SubhaloSFR', 'SubhaloVelDisp', 'SubhaloWindMass'], ill_run=1, directory='./extraction_files'):

		self.directory = directory
		self.ill_run = 1
		self.baseurl = 'http://www.illustris-project.org/api/Illustris-1/files/groupcat'

		# Illustris-1 has two missing snapshots (53, 55).
		self.snaps_to_skip = snaps_to_skip

		# the first three keys need to be Snapshot, SubhaloID, and SubhaloLenType
		self.keys = ['Snapshot', 'SubhaloID', 'SubhaloLenType'] + additional_keys

		# If the download timed out previously, figure out where it left off.
		# If not start with first snapshot.
		if 'subs_with_bhs.hdf5' in os.listdir('./extraction_files'):
			with h5py.File(self.directory + 'subs_with_bhs.hdf5', 'r') as f:
				max_snap = np.asarray(f['Snapshot'][:]).max()

			# if the maximum snap (135) is in there, move on.
			if max_snap == 135:
				print("get groupcat file info already complete")
				self.needed = False

			self.start_snap = max_snap + 1
			self.needed = True

		else:
			self.start_snap = first_snap_with_bhs
			self.needed = True

	def download_and_add_file_info(self):
		"""
		Downloads group catalog files for the information desired. It then concatenates all this data into ``subs_with_bhs.hdf5``
		"""

		# initialize output dict
		out = {key: [] for key in self.keys}

		for snap in np.arange(self.start_snap, 136):

			# skip bad snapshots (53, 55 in ill1)
			if snap in self.snaps_to_skip:
				continue

			# if any data has been output to file, gather that data into dict.
			# This method is used in case downloads time out in the middle of the run.
			if snap > self.start_snap:
				with h5py.File(self.directory + 'subs_with_bhs.hdf5', 'r') as f_out:
					for key in self.keys:
						out[key] = [f_out[key][:]]

			# download next snapshot group catalog files
			base = self.baseurl + '-%i/' % snap
			print('start download of snapshot', snap)
			for key in self.keys:
				# Get these quantities below
				if key == 'SubhaloID' or key == 'Snapshot':
					continue

				# download groupcat in quantity desired
				fp = get(base + '?Subhalo=' + key)

				# gather data from this dataset
				with h5py.File(fp, 'r') as f:
					# Run through SubgaloLenType first.
					# This keeps only subhalos with black holes in them.
					# We add subhalo and snapshot info with this.
					if key == 'SubhaloLenType':
						trans = f['Subhalo'][key][:]
						inds_keep = np.where(trans[:, 5] > 0)[0]
						out['SubhaloID'].append(inds_keep)
						out['Snapshot'].append(np.full((len(inds_keep)), snap))

					# add quantity to output dict
					out[key].append(f['Subhalo'][key][:][inds_keep])

				# remove downloaded file
				os.remove(fp)

			print('end downloads', snap)

			# move last dataset and store as backup
			if snap > self.start_snap:
				os.rename(self.directory + 'subs_with_bhs.hdf5', self.directory + 'old_subs_with_bhs.hdf5')

			# write new file with updated data.
			with h5py.File(self.directory + 'subs_with_bhs.hdf5', 'w') as f:
				for key in self.keys:
					out[key] = np.concatenate(out[key])
					f.create_dataset(key, data=out[key], dtype=out[key].dtype.name, chunks=True, compression='gzip', compression_opts=9)

			print(snap, 'groupcat completed')

		# if all files have been dowloaded, delete backup file
		if snap == 135:
			os.remove(self.directory + 'old_subs_with_bhs.hdf5')
		return
