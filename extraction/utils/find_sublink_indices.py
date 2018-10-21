"""
MAIN PURPOSE: append indices for SubhaloID and DescendantID to ``sublink_short.hdf5`` file for easy descendant searching.
"""
import numpy as np
import h5py


class SublinkIndexFind:
	"""
	SublinkIndexFind finds the associated index in the these datasets of the SubhaloID and corresponding DescendantID. These indices represent the row of the subhaloID which are done in numerical order and in line with the sublink tree they belong to. Examine the sublink datasets to see this trend. This makes finding descendants much easier. Right now it is only set to do the first 6 files because beyond that, there are no black holes present. Need to use ``sublink_short_i.hdf5`` because of time contraints of searching the hole data set.

		attributes:
			:param	num_files - (int) - number of ``sublink_short_i.hdf5`` files
			:param	directory - (str) - directory to store final files in

			needed - (bool) - does this code need to run

		methods:
			find_indices
	"""

	def __init__(self, num_files=6, directory='./extraction_files'):
		self.directory = directory
		self.num_files = num_files

		# check if needed
		with h5py.File(self.directory + 'sublink_short.hdf5', 'r') as f:
			keys = list(f)
			if 'Descendant_index' in keys:
				print('Descendant indices already added to sublink_short.')
				self.needed = False

			else:
				self.needed = True

	def find_indices(self):

		# initialize lists to hold indices
		desc_inds = []
		subhaloID_inds = []

		# initialize last index for a unique tree
		last_ind = 0
		for j in range(self.num_files):
			with h5py.File(self.directory + 'sublink_short_%i.hdf5' % j, 'r') as f:

				# find unique trees in this file
				uni, inds = np.unique(f['TreeID'][:], return_index=True)
				length = len(uni)
				print('File:', j, 'Length:', length)
				for i in range(length):

					# finds the difference between the initial subhalo id and the last index assigned
					# all the subhalos in the same tree are incremental
					subtract_inds = f['SubhaloID'][inds[i]] - last_ind

					# figure out the index change from the subhalo ids and append indices to list
					# if descendent is -1, then just put -1 rather than the index
					# last tree in current datset
					if i + 1 == length:
						diff = len(f['SubhaloID'][:]) - inds[i]

						desc_inds.append((f['DescendantID'][inds[i]::] - subtract_inds)*(f['DescendantID'][inds[i]::] != -1) + -1*(f['DescendantID'][inds[i]::] == -1))
						subhaloID_inds.append(f['SubhaloID'][inds[i]::] - subtract_inds)

					# every tree except the last one
					else:
						diff = inds[i+1] - inds[i]
						desc_inds.append((f['DescendantID'][inds[i]:inds[i+1]] - subtract_inds)*(f['DescendantID'][inds[i]:inds[i+1]] != -1) + -1*(f['DescendantID'][inds[i]:inds[i+1]] == -1))
						subhaloID_inds.append(f['SubhaloID'][inds[i]:inds[i+1]] - subtract_inds)

					# update the last index from assigning indices to this tree.
					last_ind = last_ind + diff

					print('File:', j, 'Num:', i, 'finished')

			print('File:', j, 'finished\n\n')

		# concatenate index lists and append to full ``sublink_short.hdf5``
		desc_inds = np.concatenate(desc_inds)
		subhaloID_inds = np.concatenate(subhaloID_inds)
		with h5py.File(self.directory + 'sublink_short.hdf5', 'a') as f:
			print('start descendants append')
			f.create_dataset('Descendant_index', data=desc_inds, dtype=desc_inds.dtype.name, chunks=True, compression='gzip', compression_opts=9)

			print('start subhaloID append')
			f.create_dataset('Subhalo_index', data=subhaloID_inds, dtype=subhaloID_inds.dtype.name, chunks=True, compression='gzip', compression_opts=9)

		return
