"""
This file controls the Illustris black hole extraction and filtering process. 
"""

import os
import argparse

from utils.prepare_sublink_trees import PrepSublink
from utils.get_group_subs import GetGroupSubs
from utils.find_sublink_indices import SublinkIndexFind
from utils.find_bhs import LocateBHs
from utils.sub_partIDs_in_mergs import SubPartIDs
from utils.test_good_bad_mergers import FindBadBlackHoles, TestGoodBadMergers
from utils.get_subhalos_for_download import FindSubhalosForSearch
from utils.download_needed import DownloadNeeded


class MainProcess:
	"""
	MainProcess contains and runs each major piece associated with the Illustris black hole extraction and filtering process. 

		attributes:
			:param	directory - (str) - directory to work in

		methods:
			sublink_extraction
			get_group_subs
			find_sublink_indices
			gather_black_hole_information
			sub_partIDs_in_mergs
			test_good_bad_mergers
			get_subhalos_for_download
			download_needed
	"""

	def __init__(self, directory):
		self.directory = directory

		try:
			os.listdir(directory)

		except FileNotFoundError:
			os.mkdir(directory)

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
			Find all mergers where the low mass black hole ID lives on. Update IDs for all future events so that the more massive bh ID lives on. This adds columns with new IDs to ``bhs_mergers_new.hdf5``, ``bhs_all_new.hdf5``, and ``bhs_details_new.hdf5``. This process works very well but is not perfect. Where this fails, we still keep black holes in the end that are 10M_{seed} (>10^6) (see Kelley et al 2017 for this cut). 
		"""
		print('\nStart substituting IDs for continuity.')

		sub_partIDs_in_mergs_kwargs = {
			'ill_run':3, 
			'directory':self.directory,
			'run_details':False,
		}

		sub_ids = SubPartIDs(**sub_partIDs_in_mergs_kwargs)
		if sub_ids.mergers_needed or sub_ids.all_needed or sub_ids.details_needed:
			sub_ids.find_necessary_switches()

		if sub_ids.mergers_needed:
			sub_ids.add_new_columns_to_merger_file()

		if sub_ids.all_needed:
			sub_ids.add_new_ids_to_all_bhs_file()

		#by default the details file is not done
		if sub_ids.details_needed:
			sub_ids.add_new_ids_to_details_file()


		print('Finished substituting IDs for continuity and created ``bhs_mergers_new.hdf5`` and ``bhs_details_new.hdf5`` files.\n')

		return

	def test_good_bad_mergers(self):
		"""
		##### Find Good/Bad Mergers #####
			First: we find all bad black holes resulting from fly-by encounters of host galaxies. Bad black holes are spawned by a host galaxy after it loses its original black hole in the fly-by encounter. See the documentation for `find_bad_black_holes.py` for information on this procedure. This procedure is the key step to accessing the near-seed mass black holes. 

			Second: combine these black holes with bhs that exist for less than two snapshots and have masses less than 10^6. This then produces the ``good_mergers.txt`` file detailed the good mergers by black hole standards. 
		"""
		print('\nStart finding good/bad mergers.')

		find_bad_black_holes_kwargs = {
			'directory':self.directory
		}

		bad_bhs = FindBadBlackHoles(**find_bad_black_holes_kwargs)
		if bad_bhs.needed:
			bad_bhs.search_bad_black_holes()

		test_good_bad_mergers_kwargs = {
			'directory':self.directory
		}

		good_or_bad_mergers = TestGoodBadMergers(**test_good_bad_mergers_kwargs)
		if good_or_bad_mergers.needed:
			good_or_bad_mergers.test_mergers()

		print('Finished finding good/bad mergers and created file ``good_mergers.txt``.\n')
		return

	def get_subhalos_for_download(self):
		"""
		##### Gather Subhalos for Download #####
			Gather all the subhalos associated with black hole mergers, check if they have a required resolution, and then create a file to guide the downloading process. 
		"""
		print('\nStart gathering subhalos for download.')

		get_subhalos_for_download_kwargs = {
			'directory':self.directory, 
			'skip_snaps':[53,55], 
			'use_second_sub_back':False,
		}

		gather_subs = FindSubhalosForSearch(**get_subhalos_for_download_kwargs)
		if gather_subs.needed:
			gather_subs.find_subs_to_search()

		print('Finished gathering subhalos for download and created file ``snaps_and_subs_needed.txt``.\n')

		return

	def download_needed(self):
		"""
		##### Download All Needed Subhalos #####
			This downloads all the particle information needed in each subhalo to analyze merger-related host galaxies. For remnent host galaxies, we want information about stars, gas, dm, and bhs. For constituent host galaxies, we want information about bhs and stars. (bhs are not really necessary because we have this information, but the memory necessary for this is really small, so we include it for completeness.)
		"""
		print('\nStart downloading subhalos.')

		download_needed_kwargs = {
			'directory':self.directory, 
			'ill_run':1,
		}

		download = DownloadNeeded(**download_needed_kwargs)
		#this one does not check if it is needed. It downloads based on ``completed_snaps_and_subs.txt``. 
		download.download_needed_subhalos()


		print('Finished downloading subhalos.\n')

		return


def main():

	#default is to run the whole thing

	parser = argparse.ArgumentParser()
	parser.add_argument("--all", action="store_true")
	parser.add_argument("--directory", type=str, default='./extraction_files/')
	parser.add_argument("--sublink_extraction", action="store_true")
	parser.add_argument("--get_group_subs", action="store_true")
	parser.add_argument("--find_sublink_indices", action="store_true")
	parser.add_argument("--gather_black_hole_information", action="store_true")
	parser.add_argument("--sub_partIDs_in_mergs", action="store_true")
	parser.add_argument("--test_good_bad_mergers", action="store_true")
	parser.add_argument("--download_needed", action="store_true")

	args = vars(parser.parse_args())

	keys = ['sublink_extraction', 'get_group_subs', 'find_sublink_indices', 'gather_black_hole_information', 'gather_black_hole_information', 'sub_partIDs_in_mergs', 'test_good_bad_mergers', 'download_needed']

	if True not in list(args.values()) or args['all']:
		print('Running all functions')
		for key in keys:
			args[key] = True

	else:
		for key in keys:
			if args[key]:
				print('Running', key)

	main_process = MainProcess(args['directory'])
	for key in keys:
		if args[key]:
			getattr(main_process, key)()

	return




if __name__ == '__main__':
	############ ADD ARG PARSER TO PICK WHICH PART TO RUN ################

	main()
