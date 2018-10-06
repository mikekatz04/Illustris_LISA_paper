import os

from utils.prepare_sublink_trees import PrepSublink
from utils.get_group_subs import GetGroupSubs
from utils.find_sublink_indices import SublinkIndexFind
from utils.find_bhs import LocateBHs
from utils.sub_partIDs_in_mergs import SubPartIDs


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
		print('\nStart index find within subhalo_short.hdf5')

		find_sublink_indices_kwargs = {
		'num_files': 6,
		'directory':self.directory
		}

		sublink_indices = SublinkIndexFind(**find_sublink_indices_kwargs)
		if sublink_indices.needed:
			sublink_indices.find_indices()


		print('Finished index search within subhalo_short.hdf5.\n')

		return

	def gather_black_hole_information(self):
		"""
		##### Gather All Black Hole Particle Info #####
			Find all the black holes at each snapshot, gather there particle information and write them to a file ``bhs_all_new.hdf5``. This is performed by downloading snapshot chunks and group catalog chunks to efficiently search subhalos with bhs and match black hole particle IDs to there corresponding host galaxy. 
		"""
		print('\nStart finding all bhs particle information.')

		gather_black_hole_information_kwargs = {
		'ill_run':3, 
		'directory':self.directory, 
		'num_chunk_files_per_snapshot':512, 
		'num_groupcat_files':1, 
		'first_snap_with_bhs':30, 
		'skip_snaps':[53,55], 
		'max_snap':135
		}

		find_bhs = LocateBHs(**gather_black_hole_information_kwargs)
		if find_bhs.needed:
			find_bhs.download_bhs_all_snapshots()
			find_bhs.combine_black_hole_files()


		print('Finished finding black hole particle information and created ``bhs_all_new.hdf5`` file.\n')

		return

	def sub_partIDs_in_mergs(self):
		"""
		##### Substitue Particle IDs for Continuity #####
			Find all mergers where the low mass black hole ID lives on. Update IDs for all future events so that the more massive bh ID lives on. This adds columns with new IDs to ``bhs_mergers_new.hdf5``, ``bhs_all_new.hdf5``, and ``bhs_details_new.hdf5``. 
		"""
		print('\nStart substituting IDs for continuity.')

		sub_partIDs_in_mergs_kwargs = {
		'ill_run':3, 
		'directory':self.directory
		}

		sub_ids = SubPartIDs(**sub_partIDs_in_mergs_kwargs)
		if sub_ids.mergers_needed or sub_ids.all_needed or sub_ids.details_needed:
			sub_ids.find_necessary_switches()

		if sub_ids.mergers_needed:
			sub_ids.add_new_columns_to_merger_file()

		if sub_ids.all_needed:
			sub_ids.add_new_ids_to_all_bhs_file()

		if sub_ids.details_needed:
			sub_ids.add_new_ids_to_details_file()


		print('Finished substituting IDs for continuity and created ``bhs_mergers_new.hdf5`` and ``bhs_details_new.hdf5`` files.\n')

		return


def main():

	directory = './extraction_files/'
	try:
		os.listdir(directory)

	except FileNotFoundError:
		os.mkdir(directory)

	main_process = MainProcess(directory)
	#main_process.sublink_extraction()
	#main_process.get_group_subs()
	#main_process.find_sublink_indices()
	main_process.gather_black_hole_information()
	main_process.sub_partIDs_in_mergs()





if __name__ == '__main__':
	main()
