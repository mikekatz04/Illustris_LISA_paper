"""
This file controls the Illustris black hole extraction and filtering process.
"""

import os
import sys
import argparse
import warnings

from utils import (prepare_sublink_trees, get_group_subs, find_sublink_indices, find_bhs,
                   sub_part_ids, test_good_bad_mergers, get_subhalos_for_download,
                   download_needed, density_vel_disp, create_final_data)

DEF_INPUT = '/n/ghernquist/Illustris/Runs/L75n1820FP/'
# DEF_OUTPUT = "./extraction_files/"
# DEF_OUTPUT = "./fs3_extraction-files/"
DEF_OUTPUT = "./regal_extraction-files/"

DOUBLE_CHECK_RECREATE = False


class Core:

    def __init__(self, dir_output, dir_input, recreate=False, debug=False,
                 ill_run=1, max_snap=135, first_snap_with_bhs=30, skip_snaps=[53, 55]):

        dir_output = os.path.realpath(dir_output)
        dir_input = os.path.realpath(dir_input)

        if not os.path.isdir(dir_input):
            raise RuntimeError("Input dir_output '{}' does not exist!".format(dir_input))

        if not os.path.isdir(dir_output):
            if os.path.exists(dir_output):
                raise RuntimeError("Output '{}' exists but is not dir_output!".format(dir_output))

            os.mkdir(dir_output)

        self.dir_output = dir_output
        self.dir_input = dir_input
        self.ill_run = ill_run
        self.ill_run = ill_run
        self.max_snap = max_snap
        self.first_snap_with_bhs = first_snap_with_bhs
        self.skip_snaps = skip_snaps

        self.RECREATE = recreate
        self.DEBUG = debug

        return

    def path_output(self, fname):
        return os.path.join(self.dir_output, fname)

    def fname_subs_with_bhs(self):
        return self.path_output("subs_with_bhs.hdf5")

    def fname_bhs_snapshot(self, snap):
        return self.path_output('%i/%i_blackholes.hdf5' % (snap, snap))

    def fname_sublink_short(self):
        return self.path_output('sublink_short.hdf5')

    def fname_snaps_and_subs(self):
        return self.path_output('snaps_and_subs_needed.txt')

    def fname_snaps_and_subs_completed(self):
        return self.path_output('completed_snaps_and_subs.txt')

    def fname_bhs_mergers(self):
        fname = self.path_output('bhs_mergers_new.hdf5')
        return fname

    def fname_bhs_all(self):
        fname = self.path_output('bhs_all_new.hdf5')
        return fname

    def fname_bhs_details(self):
        fname = self.path_output('bhs_details_new.hdf5')
        return fname

    def fname_illustris_bh_mergers(self):
        fname = self.path_output('blackhole_mergers-ILL%i.hdf5' % self.ill_run)
        return fname

    def fname_illustris_bh_details(self):
        fname = self.path_output('blackhole_details-ILL%i.hdf5' % self.ill_run)
        return fname

    def fname_good_mergers(self):
        return self.path_output('good_mergers.txt')

    def fname_bad_mergers(self):
        return self.path_output('bad_mergers.txt')

    def fname_snap_sub_cutout(self, snap, sub):
        # '%i/%i_sub_cutouts/cutout_%i_%i.hdf5' % (snap, snap, snap, sub)
        fname = 'cutout_%i_%i.hdf5' % (snap, sub)
        sub_path = os.path.join('%i' % snap, '%i_sub_cutouts' % snap, fname)
        return self.path_output(sub_path)

    def fname_density_profiles(self):
        fname = 'density_profiles.txt'
        fname = self.path_output(fname)
        return fname

    def fname_vel_disp(self):
        fname = 'velocity_dispersions.txt'
        fname = self.path_output(fname)
        return fname

    def fname_final_data(self):
        fname = 'simulation_input_data.txt'
        fname = self.path_output(fname)
        return fname


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
            sub_part_ids
            test_good_bad_mergers
            get_subhalos_for_download
            download_needed
            density_vel_disp
            create_final_data
    """

    Prepare_Sublink_Trees = prepare_sublink_trees.Prepare_Sublink_Trees
    Get_Group_Subs = get_group_subs.Get_Group_Subs
    Find_Sublink_Indices = find_sublink_indices.Find_Sublink_Indices
    Find_BHs = find_bhs.Find_BHs
    Sub_Part_IDs = sub_part_ids.Sub_Part_IDs
    FindBadBlackHoles = test_good_bad_mergers.FindBadBlackHoles
    TestGoodBadMergers = test_good_bad_mergers.TestGoodBadMergers
    Get_Subhalos = get_subhalos_for_download.Get_Subhalos
    Download_Needed = download_needed.Download_Needed
    Dens_Vel_Disp = density_vel_disp.Density_Vel_Disp
    Create_Final_Data = create_final_data.Create_Final_Data

    # ill_run = 1
    # max_snap = 135
    # first_snap_with_bhs = 30
    # skip_snaps = [53, 55]

    def __init__(self, core):
        print(self.__class__.__name__)
        self.core = core
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

        prep_sublink = self.Prepare_Sublink_Trees(self.core, **prep_sublink_kwargs)
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

        get_groupcat = self.Get_Group_Subs(self.core, **get_group_subs_kwargs)
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

        sublink_indices = self.Find_Sublink_Indices(self.core, **find_sublink_indices_kwargs)
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

        get_bhs = self.Find_BHs(self.core, **find_bhs_kwargs)
        if get_bhs.needed:
            print("Downloading bhs all snapshots")
            get_bhs.download_bhs_all_snapshots()
            print("Combining bh files")
            get_bhs.combine_black_hole_files()

        print('Finished finding black hole particle information and created ``bhs_all_new.hdf5`` file.\n')

        return

    def sub_part_ids(self):
        """
        # Substitue Particle IDs for Continuity #
            Find all mergers where the low mass black hole ID lives on. Update IDs for all future events so that the more massive bh ID lives on. This adds columns with new IDs to ``bhs_mergers_new.hdf5``, ``bhs_all_new.hdf5``, and ``bhs_details_new.hdf5``. This process works very well but is not perfect. Where this fails, we still keep black holes in the end that are 10M_{seed} (>10^6) (see Kelley et al 2017 for this cut).
        """
        print('\nStart substituting IDs for continuity.')

        sub_part_ids_kwargs = {
            'run_details': True,
        }

        FORCE = False

        print("Sub_Part_IDs = ", self.Sub_Part_IDs)
        sub_ids = self.Sub_Part_IDs(self.core, **sub_part_ids_kwargs)

        if FORCE:
            warnings.warn("FORCING in `sub_part_ids()`")

        if FORCE or sub_ids.mergers_needed or sub_ids.all_needed or sub_ids.details_needed:
            print("find_necessary_switches()")
            sub_ids.find_necessary_switches()

        if FORCE or sub_ids.mergers_needed:
            print("add_new_columns_to_merger_file()")
            sub_ids.add_new_columns_to_merger_file()

        if FORCE or sub_ids.all_needed:
            print("add_new_ids_to_all_bhs_file()")
            sub_ids.add_new_ids_to_all_bhs_file()

        # by default the details file is not done
        if FORCE or sub_ids.details_needed:
            print("add_new_ids_to_details_file()")
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

        bad_bhs = self.FindBadBlackHoles(self.core, **find_bad_black_holes_kwargs)
        if bad_bhs.needed:
            bad_bhs.search_bad_black_holes()

        test_good_bad_mergers_kwargs = {}

        good_or_bad_mergers = self.TestGoodBadMergers(
            self.core, **test_good_bad_mergers_kwargs)
        if good_or_bad_mergers.needed:
            good_or_bad_mergers.test_mergers()

        print('Finished finding good/bad mergers and created file ``good_mergers.txt``.\n')
        return

    def get_subhalos_for_download(self):
        """
        # Gather Subhalos for Download #
            Gather all the subhalos associated with black hole mergers, check if they have a
            required resolution, and then create a file to guide the downloading process.
        """
        print('\nStart gathering subhalos for download.')

        get_subhalos_for_download_kwargs = {
            'use_second_sub_back': False,
        }

        gather_subs = self.Get_Subhalos(self.core, **get_subhalos_for_download_kwargs)
        if gather_subs.needed:
            gather_subs.find_subs_to_search()

        print('Finished gathering subhalos for download and created file'
              ' ``snaps_and_subs_needed.txt``.\n')

        return

    def download_needed(self):
        """
        # Download All Needed Subhalos #
            This downloads all the particle information needed in each subhalo to analyze merger-related host galaxies. For remnent host galaxies, we want information about stars, gas, dm, and bhs. For constituent host galaxies, we want information about bhs and stars. (bhs are not really necessary because we have this information, but the memory necessary for this is really small, so we include it for completeness.)
        """
        print('\nStart downloading subhalos.')

        download_needed_kwargs = {}

        download = self.Download_Needed(self.core, **download_needed_kwargs)
        # this one does not check if it is needed.
        # It downloads based on ``completed_snaps_and_subs.txt``.
        download.download_needed_subhalos()

        print('Finished downloading subhalos.\n')

        return

    def density_vel_disp(self):
        """
        # Calculate Density Profiles and Stellar Velocity Dispersions #
            This calculates density profiles and velocity dispersions for the mergers. It gets density profiles for all particle types in remnant black hole host galaxies. Stellar velocity dispersions are calcualted for all merger-related galaxies. If a fit does not converge, the merger is no longer considered part of our catalog.
        """
        print('\nStart calculating profiles and dispersions.')

        density_vel_disp_kwargs = {}

        dens_vel = self.Dens_Vel_Disp(self.core, **density_vel_disp_kwargs)
        # this one does not check if it is needed. downloads based on `completed_snaps_and_subs.txt`

        if dens_vel.needed:
            dens_vel.fit_main()

        print('Finished calculating profiles and dispersions and produced files ``density_profiles.txt`` and ``velocity_dispersion.txt``.\n')

        return

    def create_final_data(self):
        """
        # Create Final Dataset #
            This gathers all of the information attained in this analysis and combines the good mergers in a final dataset.
        """
        print('\nStart generating final dataset.')

        create_final_data_kwargs = {}

        final_data = self.Create_Final_Data(self.core, **create_final_data_kwargs)

        if final_data.needed:
            final_data.create_final_data()

        print('Finished generating final dataset and created file `simulation_input_data.txt`.\n')

        return


class MainProcess_Odyssey(MainProcess):

    Prepare_Sublink_Trees = prepare_sublink_trees.Prepare_Sublink_Trees_Odyssey
    Find_Sublink_Indices = find_sublink_indices.Find_Sublink_Indices_Odyssey
    Get_Group_Subs = get_group_subs.Get_Group_Subs_Odyssey
    Find_BHs = find_bhs.Find_BHs_Odyssey
    Sub_Part_IDs = sub_part_ids.Sub_Part_IDs_Odyssey
    Download_Needed = download_needed.Download_Needed_Odyssey

    '''
    def sublink_extraction(self):
        print("\t`sublink_extraction` is not needed on Odyssey")
        return
    '''

    # def find_sublink_indices(self):
    #     print("\tWARNING: skipping `find_sublink_indices` on Odyssey!")
    #     return


def main():

    # default is to run the whole thing

    parser = argparse.ArgumentParser()
    parser.add_argument("-a", "--all", action="store_true", default=False)
    parser.add_argument("-r", "--recreate", action="store_true", default=False)
    parser.add_argument("-d", "--debug", action="store_true", default=True)
    parser.add_argument("--odyssey", action="store_true", default=True)
    parser.add_argument("--dir_output", type=str, default=DEF_OUTPUT)
    parser.add_argument("--dir_input", type=str, default=DEF_INPUT)

    DEF = False

    keys = [['sublink_extraction', DEF],
            ['get_group_subs', DEF],
            ['find_sublink_indices', DEF],
            ['find_bhs', DEF],
            ['sub_part_ids', DEF],
            ['test_good_bad_mergers', DEF],
            ['get_subhalos_for_download', DEF],
            ['download_needed', DEF],
            ['density_vel_disp', DEF],
            ['create_final_data', DEF]]
    # 'gather_black_hole_information',

    for key, val in keys:
        parser.add_argument("--" + key, action='store_true', default=val)

    args = vars(parser.parse_args())

    if args['all']:
        print('Running all functions')
        for key, _ in keys:
            args[key] = True

    if args['odyssey']:
        print("Running in Odyssey mode!")
        Main_Process = MainProcess_Odyssey
    else:
        Main_Process = MainProcess

    if args['recreate'] or args['debug']:
        for key, _ in keys:
            print("{:>30s}: {}".format(key, args[key]))

    if args['recreate']:
        print("\nWARNING: running in `recreate` mode!\n")
        if DOUBLE_CHECK_RECREATE:
            arg = input("\nConfirm recreate all files Y/[N] : ").strip().lower()

            if not arg.startswith('y'):
                print("Aborting.")
                sys.exit(0)

    core = Core(args['dir_output'], args['dir_input'], args['recreate'], args['debug'])
    main_process = Main_Process(core)
    for key, _ in keys:
        if args[key]:
            if args['debug']:
                print("Running '{}'".format(key))

            getattr(main_process, key)()

    return


if __name__ == '__main__':
    main()
