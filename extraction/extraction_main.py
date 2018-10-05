import os

from utils.prepare_sublink_trees import PrepSublink
from utils.get_group_subs import GetGroupSubs
from utils.find_sublink_indices import SublinkIndexFind


class MainProcess:

	def __init__(self, directory):
		self.directory = directory

	def sublink_extraction(self):
		"""
		###### Sublink Extraction #######
		We first download the sublink trees and extract only the necessary information to conserve memory. The extracted files are kept separate as ``sublink_short_i.hdf5`` for the ith sublink file. There is also ``sublink_short.hdf5`` file that combines all this information. The code is designed to check for these files and skip if needed. 
		"""

		print('\n\nStart Preparing Sublink Files')
		prep_sublink_kwargs = {
			'num_files':2,
			'keys':['DescendantID', 'SnapNum', 'SubfindID', 'SubhaloID', 'SubhaloLenType', 'SubhaloMass','SubhaloMassInHalfRad', 'SubhaloMassType', 'TreeID','SubhaloSFR'],
			'directory':self.directory,
			'ill_run':3,
		}

		prep_sublink = PrepSublink(**prep_sublink_kwargs)
		if prep_sublink.needed:
			prep_sublink.download_and_convert_to_short()
			prep_sublink.combine_sublink_shorts()

		print('Finished Preparing Sublink Files\n')
		return

	def get_group_subs(self):
		"""
		###### Group Catalog Information #####
		Get group catalog information for all the subhalos with black holes in them. Code is designed to download from Illustris server, at snapshot intervals. It picks back up where it left off if the download times out. 
		"""
		print('\nStart group catalog extraction for subhalos with black holes.')

		get_group_subs_kwargs = {
		'first_snap_with_bhs':31, 
		'snaps_to_skip':[53,55], 
		'additional_keys':['SubhaloCM', 'SubhaloMassType', 'SubhaloPos', 'SubhaloSFR', 'SubhaloVelDisp', 'SubhaloWindMass'], 
		'ill_run':3, 
		'directory':self.directory
		}

		get_groupcat = GetGroupSubs(**get_group_subs_kwargs)
		if get_groupcat.needed:
			get_groupcat.download_and_add_file_info()


		print('Finished extracting subhalo information from group catalog files.\n')

		return

	def find_sublink_indices(self):
		"""
		###### Find Sublink Indices #####
			Append index information into the ``sublink_short.hdf5`` dataset for the SubhaloID and DescendantID. This allows for fast descendant searches.
		"""
		print('\nStart index find with subhalo_short.hdf5')

		find_sublink_indices_kwargs = {
		'num_files': 6,
		'directory':self.directory
		}

		sublink_indices = SublinkIndexFind(**find_sublink_indices_kwargs)
		if sublink_indices.needed:
			sublink_indices.find_indices()


		print('Finished index search within subhalo_short.hdf5.\n')

		return



def main():

	directory = './extraction_files/'
	try:
		os.listdir(directory)

	except FileNotFoundError:
		os.mkdir(directory)

	main_process = MainProcess(directory)
	main_process.sublink_extraction()
	main_process.get_group_subs()
	main_process.find_sublink_indices()






if __name__ == '__main__':
	main()
