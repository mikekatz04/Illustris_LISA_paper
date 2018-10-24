"""
This file controls the Illustris black hole extraction and filtering process.
"""

import os
import argparse

# from utils.prepare_sublink_trees import PrepSublink
# from utils.get_group_subs import GetGroupSubs
# from utils.find_sublink_indices import SublinkIndexFind
# from utils.find_bhs import LocateBHs

# from utils.sub_partIDs_in_mergs import SubPartIDs
# from utils.test_good_bad_mergers import FindBadBlackHoles, TestGoodBadMergers
# from utils.get_subhalos_for_download import FindSubhalosForSearch
# from utils.download_needed import DownloadNeeded
# from utils.density_vel_disp_of_subs import DensityProfVelDisp
# from utils.create_final_data import CreateFinalDataset

from utils import (prepare_sublink_trees, get_group_subs, find_sublink_indices, find_bhs,
                   sub_partIDs_in_mergs, test_good_bad_mergers, get_subhalos_for_download,
                   download_needed)


class MainProcess:
    """
    MainProcess contains and runs each major piece associated with the Illustris black hole extraction and filtering process.

        attributes:
            :param  dir_output - (str) - dir_output to work in

        methods:
            sublink_extraction
            get_group_subs
            find_sublink_indices
            find_bhs
            sub_partIDs_in_mergs
            test_good_bad_mergers
            get_subhalos_for_download
            download_needed
            density_vel_disp_of_subs
            create_final_data
    """

    PrepSublink = prepare_sublink_trees.PrepSublink
    GetGroupSubs = get_group_subs.GetGroupSubs
    SublinkIndexFind = find_sublink_indices.SublinkIndexFind
    LocateBHs = find_bhs.LocateBHs
    SubPartIDs = sub_partIDs_in_mergs.SubPartIDs
    FindBadBlackHoles = test_good_bad_mergers.FindBadBlackHoles
    TestGoodBadMergers = test_good_bad_mergers.TestGoodBadMergers
    FindSubhalosForSearch = get_subhalos_for_download.FindSubhalosForSearch
    DownloadNeeded = download_needed.DownloadNeeded

    ill_run = 1
    max_snap = 135
    first_snap_with_bhs = 30
    skip_snaps = [53, 55]

    def __init__(self, dir_output, dir_input=None):
        print(self.__class__.__name__)
        self.dir_input = dir_input
        self.dir_output = dir_output

        if not os.path.isdir(dir_input):
            raise RuntimeError("Input dir_output '{}' does not exist!".format(dir_input))

        if not os.path.isdir(dir_output):
            if os.path.exists(dir_output):
                raise RuntimeError("Output '{}' exists but is not dir_output!".format(dir_output))

            os.mkdir(dir_output)

            return

    def sublink_extraction(self):
        """
        # Sublink Extraction #
        We first download the sublink trees and extract only the necessary information to conserve memory. The extracted files are kept separate as ``sublink_short_i.hdf5`` for the ith sublink file. There is also ``sublink_short.hdf5`` file that combines all this information. The code is designed to check for these files and skip if needed.
        """

        print('\n\nStart Preparing Sublink Files')
        prep_sublink_kwargs = {
            'num_files': 2,
            'keys': ['DescendantID', 'SnapNum', 'SubfindID', 'SubhaloID', 'SubhaloLenType', 'SubhaloMass', 'SubhaloMassInHalfRad', 'SubhaloMassType', 'TreeID', 'SubhaloSFR'],
        }

        prep_sublink = self.PrepSublink(self, **prep_sublink_kwargs)
        if prep_sublink.needed:
            prep_sublink.download_and_convert_to_short()
            prep_sublink.combine_sublink_shorts()

        print('Finished Preparing Sublink Files\n')
        return

    def get_group_subs(self):
        """
        # Group Catalog Information #
        Get group catalog information for all the subhalos with black holes in them. Code is designed to download from Illustris server, at snapshot intervals. It picks back up where it left off if the download times out. Returns ``subs_with_bhs.hdf5``
        """

        print('\nStart group catalog extraction for subhalos with black holes.')

        get_group_subs_kwargs = {
            'additional_keys': ['SubhaloCM', 'SubhaloMassType', 'SubhaloPos', 'SubhaloSFR', 'SubhaloVelDisp', 'SubhaloWindMass'],
        }

        get_groupcat = self.GetGroupSubs(self, **get_group_subs_kwargs)
        if get_groupcat.needed:
            get_groupcat.download_and_add_file_info()

        print('Finished extracting subhalo information from group catalog files.\n')

        return

    def find_sublink_indices(self):
        """
        # Find Sublink Indices #
            Append index information into the ``sublink_short.hdf5`` dataset for the SubhaloID and DescendantID. This allows for fast descendant searches.
        """
        print('\nStart index find within subhalo_short.hdf5')

        find_sublink_indices_kwargs = {
            'num_files': 6,
        }

        sublink_indices = self.SublinkIndexFind(self, **find_sublink_indices_kwargs)
        if sublink_indices.needed:
            sublink_indices.find_indices()

        print('Finished index search within subhalo_short.hdf5.\n')

        return

    def find_bhs(self):
        """
        # Gather All Black Hole Particle Info #
            Find all the black holes at each snapshot, gather there particle information and write them to a file ``bhs_all_new.hdf5``. This is performed by downloading snapshot chunks and group catalog chunks to efficiently search subhalos with bhs and match black hole particle IDs to there corresponding host galaxy.
        """
        print('\nStart finding all bhs particle information.')

        find_bhs_kwargs = {
            'num_chunk_files_per_snapshot': 512,
            'num_groupcat_files': 1,
        }

        get_bhs = self.LocateBHs(self, **find_bhs_kwargs)
        if get_bhs.needed:
            get_bhs.download_bhs_all_snapshots()
            get_bhs.combine_black_hole_files()

        print('Finished finding black hole particle information and created ``bhs_all_new.hdf5`` file.\n')

        return

    def sub_partIDs_in_mergs(self):
        """
        # Substitue Particle IDs for Continuity #
            Find all mergers where the low mass black hole ID lives on. Update IDs for all future events so that the more massive bh ID lives on. This adds columns with new IDs to ``bhs_mergers_new.hdf5``, ``bhs_all_new.hdf5``, and ``bhs_details_new.hdf5``. This process works very well but is not perfect. Where this fails, we still keep black holes in the end that are 10M_{seed} (>10^6) (see Kelley et al 2017 for this cut).
        """
        print('\nStart substituting IDs for continuity.')

        sub_partIDs_in_mergs_kwargs = {
            'run_details': False,
        }

        sub_ids = self.SubPartIDs(self, **sub_partIDs_in_mergs_kwargs)
        if sub_ids.mergers_needed or sub_ids.all_needed or sub_ids.details_needed:
            sub_ids.find_necessary_switches()

        if sub_ids.mergers_needed:
            sub_ids.add_new_columns_to_merger_file()

        if sub_ids.all_needed:
            sub_ids.add_new_ids_to_all_bhs_file()

        # by default the details file is not done
        if sub_ids.details_needed:
            sub_ids.add_new_ids_to_details_file()

        print('Finished substituting IDs for continuity and created ``bhs_mergers_new.hdf5`` and ``bhs_details_new.hdf5`` files.\n')

        return

    def test_good_bad_mergers(self):
        """
        # Find Good/Bad Mergers #
            First: we find all bad black holes resulting from fly-by encounters of host galaxies. Bad black holes are spawned by a host galaxy after it loses its original black hole in the fly-by encounter. See the documentation for `test_good_bad_mergers.py` for information on this procedure. This procedure is the key step to accessing the near-seed mass black holes.

            Second: black holes that exist for less than two snapshots and have masses less than 10^6 are also considered bad black holes.

            Afte these two steps, the filtered set of good mergers is output to ``good_mergers.txt`` file detailing the good mergers by black hole standards.
        """
        print('\nStart finding good/bad mergers.')

        find_bad_black_holes_kwargs = {}

        bad_bhs = self.FindBadBlackHoles(self, **find_bad_black_holes_kwargs)
        if bad_bhs.needed:
            bad_bhs.search_bad_black_holes()

        test_good_bad_mergers_kwargs = {}

        good_or_bad_mergers = self.TestGoodBadMergers(self, **test_good_bad_mergers_kwargs)
        if good_or_bad_mergers.needed:
            good_or_bad_mergers.test_mergers()

        print('Finished finding good/bad mergers and created file ``good_mergers.txt``.\n')
        return

    def get_subhalos_for_download(self):
        """
        # Gather Subhalos for Download #
            Gather all the subhalos associated with black hole mergers, check if they have a required resolution, and then create a file to guide the downloading process.
        """
        print('\nStart gathering subhalos for download.')

        get_subhalos_for_download_kwargs = {
            'use_second_sub_back': False,
        }

        gather_subs = self.FindSubhalosForSearch(self, **get_subhalos_for_download_kwargs)
        if gather_subs.needed:
            gather_subs.find_subs_to_search()

        print('Finished gathering subhalos for download and created file ``snaps_and_subs_needed.txt``.\n')

        return

    def download_needed(self):
        """
        # Download All Needed Subhalos #
            This downloads all the particle information needed in each subhalo to analyze merger-related host galaxies. For remnent host galaxies, we want information about stars, gas, dm, and bhs. For constituent host galaxies, we want information about bhs and stars. (bhs are not really necessary because we have this information, but the memory necessary for this is really small, so we include it for completeness.)
        """
        print('\nStart downloading subhalos.')

        download_needed_kwargs = {}

        download = self.DownloadNeeded(self, **download_needed_kwargs)
        # this one does not check if it is needed. It downloads based on ``completed_snaps_and_subs.txt``.
        download.download_needed_subhalos()

        print('Finished downloading subhalos.\n')

        return

    def density_vel_disp_of_subs(self):
        """
        # Calculate Density Profiles and Stellar Velocity Dispersions #
            This calculates density profiles and velocity dispersions for the mergers. It gets density profiles for all particle types in remnant black hole host galaxies. Stellar velocity dispersions are calcualted for all merger-related galaxies. If a fit does not converge, the merger is no longer considered part of our catalog.
        """
        print('\nStart calculating profiles and dispersions.')

        density_vel_disp_of_subs_kwargs = {
            'dir_output': self.dir_output,
        }

        dens_vel = DensityProfVelDisp(**density_vel_disp_of_subs_kwargs)
        # this one does not check if it is needed. It downloads based on ``completed_snaps_and_subs.txt``.

        if dens_vel.needed:
            dens_vel.fit_main()

        print('Finished calculating profiles and dispersions and produced files ``density_profilesl.txt`` and ``velocity_dispersion.txt``.\n')

        return

    def create_final_data(self):
        """
        # Create Final Dataset #
            This gathers all of the information attained in this analysis and combines the good mergers in a final dataset.
        """
        print('\nStart generating final dataset.')

        create_final_data_kwargs = {
            'dir_output': self.dir_output,
        }

        final_data = CreateFinalDataset(**create_final_data_kwargs)
        # this one does not check if it is needed. It downloads based on ``completed_snaps_and_subs.txt``.

        if final_data.needed:
            final_data.create_final_data()

        print('Finished generating final dataset and created file ``simulation_input_data.txt``.\n')

        return


class MainProcess_Odyssey(MainProcess):

    GetGroupSubs = get_group_subs.GetGroupSubs_Odyssey
    LocateBHs = find_bhs.LocateBHs_Odyssey

    def sublink_extraction(self):
        print("\t`sublink_extraction` is not needed on Odyssey")
        return

    def find_sublink_indices(self):
        print("\tWARNING: skipping `find_sublink_indices` on Odyssey!")
        return


def main():

    # default is to run the whole thing

    parser = argparse.ArgumentParser()
    parser.add_argument("--all", action="store_true", default=True)
    parser.add_argument("--odyssey", action="store_true", default=True)
    parser.add_argument("--dir_output", type=str, default='./extraction_files/')
    parser.add_argument("--dir_input", type=str, default='/n/ghernquist/Illustris/Runs/L75n1820FP/')

    parser.add_argument("--sublink_extraction", action="store_true")
    parser.add_argument("--get_group_subs", action="store_true")
    parser.add_argument("--find_sublink_indices", action="store_true")
    parser.add_argument("--find_bhs", action="store_true")
    parser.add_argument("--sub_partIDs_in_mergs", action="store_true")
    parser.add_argument("--test_good_bad_mergers", action="store_true")
    parser.add_argument("--download_needed", action="store_true")
    parser.add_argument("--density_vel_disp_of_subs", action="store_true")
    parser.add_argument("--create_final_data", action="store_true")

    args = vars(parser.parse_args())

    keys = ['sublink_extraction', 'get_group_subs', 'find_sublink_indices', 'find_bhs', 'sub_partIDs_in_mergs', 'test_good_bad_mergers', 'get_subhalos_for_download', 'download_needed', 'density_vel_disp_of_subs', 'create_final_data']  # 'gather_black_hole_information',

    if True not in list(args.values()) or args['all']:
        print('Running all functions')
        for key in keys:
            args[key] = True

    # else:
    #     for key in keys:
    #         if args[key]:
    #             print('Running', key)

    if args['odyssey']:
        Main_Process = MainProcess_Odyssey
    else:
        Main_Process = MainProcess

    main_process = Main_Process(args['dir_output'], dir_input=args['dir_input'])
    for key in keys:
        if args[key]:
            print("Running '{}'".format(key))
            getattr(main_process, key)()

    return


if __name__ == '__main__':
    main()
