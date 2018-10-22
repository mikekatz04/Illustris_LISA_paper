"""
MAIN PURPOSE: download sublink files and combine them to create ``sublink_short_i.hdf5`` and ``sublink_short.hdf5`` files.
"""

import numpy as np
import h5py
import os

from utils.generalfuncs import get


class PrepSublink:
	"""
	PrepSublink downloads the necessary sublink files from the Illustris server. It then moves the classes we are interested in to a separate file to preserve memory. It then deletes the original file.

		attributes:
			:param	num_files - (int) - number of sublink files to download starting at zero. This is used because not all files have subhalos with black holes.
			:param	keys - list of (str) - keys of interest
			:param	dir_output - (str) - dir_output to store final files in
			:param	ill_run - (int) - integer representing illustris run

			needed - (bool) - does this code need to run


		methods:
			download_and_convert_to_short
			combine_sublink_shorts
	"""

	def __init__(self, num_files=6, keys=['DescendantID', 'SnapNum', 'SubfindID', 'SubhaloID', 'SubhaloLenType', 'SubhaloMass', 'SubhaloMassInHalfRad', 'SubhaloMassType', 'TreeID', 'SubhaloSFR'], dir_output='./extraction_files/', ill_run=1):

		self.base_url = "http: //www.illustris-project.org/api/Illustris-%i/" % ill_run

		self.num_files, self.keys, self.dir_output = num_files, keys, dir_output

		num_files_complete = 0
		for num in range(self.num_files):
			if 'sublink_short_%i.hdf5' % num in os.listdir(self.dir_output):
				num_files_complete += 1

		if num_files_complete == self.num_files:
			self.needed = False
			print('all sublink_short files already attained')

		else:
			self.needed = True

	def download_and_convert_to_short(self):
		"""
		downloads the sublink files and converts them to a corresponding short file. These are needed for conversion of decsendant IDs to indexes.
		"""
		for num in range(self.num_files):
			# check if this file is done
			if 'sublink_short_%i.hdf5' % num in os.listdir(self.dir_output):
				print('sublink_short_%i.hdf5' % num, 'already downloaded.')
				continue

			# check if we have the downloaded file left over. If not, download it.
			if 'tree_extended.%i.hdf5' % num not in os.listdir(self.dir_output):
				downloaded_file = get(self.base_url + 'files/sublink.%i.hdf5' % num)
				os.rename(downloaded_file, self.dir_output + 'sublink.%i.hdf5' % num)

			# write quantities of interest to short version.
			with h5py.File(self.dir_output + 'sublink.%i.hdf5' % num, 'r') as old_sublink_file:
				with h5py.File(self.dir_output + 'sublink_short_%i.hdf5' % num, 'w') as new_sublink_file:

					for key in self.keys:
						out = old_sublink_file[key][:]
						new_sublink_file.create_dataset(key, data=out, dtype=out.dtype.name, chunks=True, compression='gzip', compression_opts=9)

			# delete larger dataset
			os.remove(self.dir_output + 'sublink.%i.hdf5' % num)
			print('sublink_short_%i.hdf5' % num, 'complete.')

		return

	def combine_sublink_shorts(self):
		"""
		Combines all short sublink files into a combined sublink short file.
		"""

		# check if this is already done
		if 'sublink_short.hdf5' in os.listdir(self.dir_output):
			print('sublink_short.hdf5 (combined data) already in folder.')
			return

		# initialize dict to hold all data before output
		out_dict = {key: [] for key in self.keys}

		# gather all the data from the numbered short files
		for num in range(self.num_files):
			small_file = h5py.File(self.dir_output + 'sublink_short_%i.hdf5' % num, 'r')
			for key in self.keys:
				out_dict[key].append(small_file[key][:])

		# concatenate all data
		out_dict = {key: np.concatenate(out_dict[key], axis=0) for key in self.keys}

		# write to overall short file
		with h5py.File(self.dir_output + 'sublink_short.hdf5', 'w') as combined_sublink:
			for key in self.keys:
				combined_sublink.create_dataset(key, data=out_dict[key], dtype=out_dict[key].dtype.name, chunks=True, compression='gzip', compression_opts=9)

		print('sublink_short.hdf5 complete (combined data)')
		return
