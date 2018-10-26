"""
MAIN PURPOSE: subsitute indices into sublink merger trees to make descending a tree easier and faster
"""

import os
import h5py
import numpy as np

from utils.generalfuncs import get
from utils import SubProcess


class SubPartIDs(SubProcess):
    """
    SubPartIDs is the most pivotal change to the data for better analysis. In the Illustris simulation, when two black hole particles are merged, the bh particle ID that lives on is chosen at random. This causes problems with continuity in following a black hole amongst other issues. In this code, the larger mass black holes is chosen to live on. Therefore, we locate all mergers that need an adjustment. This adjustment is applied to merger, details (default is to not do details catalog), and all bhs files for any time after the merger.

        attributes:
            :param  ill_run - (int) - illustris run to use
            :param  dir_output - (str) - dir_output to work out of
            :param  run_details - (bool) - run details catalog through this process. Default is False

            mergers_needed - (bool) - need to sub ids in merger file
            all_needed - (bool) - need to sub ids in all bhs file
            details_needed - (bool) - need to sub ids in details file
            change1, change2, change3 - (array) - information on merger ids that need to be changed

        methods:
            download_original_bh_merger_file
            find_necessary_switches
            add_new_columns_to_merger_file
            add_new_ids_to_all_bhs_file
            download_details_file
            add_new_ids_to_details_file
    """

    def __init__(self, core, run_details=False):
        super().__init__(core)

        # see if the new columns are added and if the files exist
        fname_mergers = self.core.fname_bhs_mergers()
        if os.path.exists(fname_mergers):
            with h5py.File(fname_mergers, 'r') as f:
                if 'mass_in_new' in list(f):
                    self.mergers_needed = False
                else:
                    self.mergers_needed = True

        else:
            self.mergers_needed = True

        # bhs_all_new.hdf5 should be in the folder from `find_bhs.py`
        fname_all = self.core.fname_bhs_all()
        with h5py.File(fname_all, 'r') as f:
            if 'ParticleIDs_new' in list(f):
                self.all_needed = False

        if run_details:
            fname_details = self.core.fname_bhs_details()
            if os.path.exists(fname_details):
                with h5py.File(fname_details, 'r') as f:
                    if 'id_new' in list(f):
                        self.details_needed = False
                    else:
                        self.details_needed = True
            else:
                self.details_needed = True
        else:
            self.details_needed = False

    def download_original_bh_merger_file(self):
        """
        Use `get` to download the original black hole merger file from the Illustris website.
        """

        # fname_illustris_mergers = os.path.join(self.dir_output, 'blackhole_mergers-ILL%i.hdf5' % self.ill_run)
        fname_illustris_mergers = self.core.fname_illustris_bh_mergers()
        if os.path.exists(fname_illustris_mergers):
            print('blackhole_mergers.hdf5 already downloaded.')
            return

        print('blackhole_mergers.hdf5 -> beginning download.')

        fp = get('http://www.illustris-project.org/api/Illustris-%i/files/blackhole_mergers.hdf5' % self.ill_run)

        # move and rename file in correct dir_output
        os.rename(fp, fname_illustris_mergers)

        print('blackhole_mergers-ILL%i.hdf5 -> finished download.' % self.ill_run)
        return

    def find_necessary_switches(self):
        """
        Locate all of the mergers where the IDs need to be switched.
        """

        # download the original file if it is not in folder.
        fname_mergers = self.core.fname_bhs_mergers()
        # WARNING: I think something is wrong here, wrong file being downloaded here?
        if not os.path.exists(fname_mergers):
            self.download_original_bh_merger_file()

        with h5py.File(fname_mergers, 'r') as f_merg:

            # get original quantities
            mass_in = f_merg['mass_in'][:]
            mass_out = f_merg['mass_out'][:]
            time = f_merg['time'][:]
            snap = f_merg['snapshot'][:]

            masses = np.array([mass_in, mass_out]).T

            partIDs_in, partIDs_out = f_merg['id_in'][:], f_merg['id_out'][:]

            num_bad = 0

            # populate the change list with mergers where the ID that lives on is the smaller black hole
            change = []
            for m in range(len(mass_in)):
                out_val = np.argmax(masses[m])
                if out_val == 0:
                    num_bad += 1
                    change.append((m, partIDs_out[m], partIDs_in[m], time[m], snap[m]))

            # populate structured arrays for easy reference to each quantity
            # we make three copies because we will fill adjust these arrays as we fill each file.
            self.change1 = np.array(change, dtype=[('merger', '<i4'), ('id_remove', np.dtype('uint64')), ('id_keep', np.dtype('uint64')), ('time', '<f4'), ('snap', '<i4')])

            self.change2 = np.array(change, dtype=[('merger', '<i4'), ('id_remove', np.dtype('uint64')), ('id_keep', np.dtype('uint64')), ('time', '<f4'), ('snap', '<i4')])

            self.change3 = np.array(change, dtype=[('merger', '<i4'), ('id_remove', np.dtype('uint64')), ('id_keep', np.dtype('uint64')), ('time', '<f4'), ('snap', '<i4')])

            print('Number of mergers to fix:', num_bad)
            return

    def add_new_columns_to_merger_file(self):
        """
        Fix the mergers in ``bhs_mergers_new.hdf5``.
        """
        # get original data
        fname_mergers = self.core.fname_bhs_mergers()
        with h5py.File(fname_mergers, 'r') as f_merg:
            mass_in = f_merg['mass_in'][:]
            mass_out = f_merg['mass_out'][:]
            time = f_merg['time'][:]
            # snap = f_merg['snapshot'][:]

            masses = np.array([mass_in, mass_out]).T

            partIDs_in, partIDs_out = f_merg['id_in'][:], f_merg['id_out'][:]

        change = self.change1

        # these arrays will be searched for changes
        partIDs = np.array([partIDs_in, partIDs_out]).T
        time = np.array([time, time]).T

        print('Number of mergers to change:', len(change))
        for i in range(len(change)):

            # find all future merger indices that need fixing.
            inds_fix = np.where((partIDs == change['id_remove'][i]) & (time > change['time'][i]))

            # fix this specific merger
            ind_switch = change['merger'][i]

            # this changes all future IDs of the bad black hole to the good black hole
            # nothing with mass happens here
            if len(inds_fix) != 0:
                partIDs[inds_fix] = change['id_keep'][i]

            # fix particleIDs and mass for the merger that needs to be fixed
            partIDs[ind_switch][0] = change['id_remove'][i]
            partIDs[ind_switch][1] = change['id_keep'][i]

            trans_out_keep = masses[ind_switch][0]
            trans_in_remove = masses[ind_switch][1]
            masses[ind_switch][0] = trans_in_remove
            masses[ind_switch][1] = trans_out_keep

            # fix change array so that it keeps updated bad black hole ids with good ones
            # for only future iterations
            # Need to do this for both id_keep columns and id_remove. This is becuse id_remove will be in the future and will need to reflect changes from here.
            inds_fix = np.where((change['id_keep'] == change['id_remove'][i]) & (change['time'] >= change['time'][i]))[0]
            change['id_keep'][inds_fix] = change['id_keep'][i]

            inds_fix = np.where((change['id_remove'] == change['id_remove'][i]) & (change['time'] >= change['time'][i]))[0]
            change['id_remove'][inds_fix] = change['id_keep'][i]

            print(i)

        # read out
        with h5py.File(fname_mergers, 'a') as f_merg:
            f_merg['id_in_new'] = partIDs[:, 0]
            f_merg['mass_in_new'] = masses[:, 0]

            f_merg['id_out_new'] = partIDs[:, 1]
            f_merg['mass_out_new'] = masses[:, 1]

        return

    def add_new_ids_to_all_bhs_file(self):
        """
        Update all IDs in ``bhs_all_new.hdf5`` with their new IDs
        """

        # original data
        fname_all = self.core.fname_bhs_all()
        with h5py.File(fname_all, 'r') as f_all:
            partIDs_all = f_all['ParticleIDs'][:]
            snap_all = f_all['Snapshot'][:]

        # same process as mergers, except without having to actually fix the merger
        change = self.change2
        for i in range(len(change)):
            # inds_fix = np.where((partIDs_all == change['id_remove'][i]) & (snap_all >= change['snap'][i]))[0]
            inds_fix = (partIDs_all == change['id_remove'][i]) & (snap_all >= change['snap'][i])

            # if there is no switch needed
            if np.count_nonzero(inds_fix) == 0:
                print('no change no change')

            partIDs_all[inds_fix] = change['id_keep'][i]

            # still need to update change array even if there is no switch needed
            # inds_fix = np.where((change['id_keep'] == change['id_remove'][i]) & (change['time'] >= change['time'][i]))[0]
            inds_fix = (change['id_keep'] == change['id_remove'][i]) & (change['time'] >= change['time'][i])
            change['id_keep'][inds_fix] = change['id_keep'][i]

            # inds_fix = np.where((change['id_remove'] == change['id_remove'][i]) & (change['time'] >= change['time'][i]))[0]
            inds_fix = (change['id_remove'] == change['id_remove'][i]) & (change['time'] >= change['time'][i])
            change['id_remove'][inds_fix] = change['id_keep'][i]

            print(i)

        # read out
        with h5py.File(fname_all, 'a') as f_all:
            f_all.create_dataset('ParticleIDs_new', data=partIDs_all, dtype=partIDs_all.dtype.name, chunks=True, compression='gzip', compression_opts=9)

        return

    def download_details_file(self):
        """
        Download blackhole details file from Illustris website.
        """

        # fname_illustris_details = os.path.join(self.dir_output, 'blackhole_details-ILL%i.hdf5' % self.ill_run)
        fname_illustris_details = self.core.fname_illustris_bh_details()
        if os.path.exists(fname_illustris_details):
            print('blackhole_details-ILL%i.hdf5 already downloaded.' % self.ill_run)
            return

        print('blackhole_details-ILL%i.hdf5 -> beginning download.' % self.ill_run)
        fp = get('http://www.illustris-project.org/api/Illustris-%i/files/blackhole_details.hdf5' % self.ill_run)
        os.rename(fp, fname_illustris_details)
        print('blackhole_details-ILL%i.hdf5 -> finished download.' % self.ill_run)
        return

    def add_new_ids_to_details_file(self):
        """
        Update IDs in ``bhs_details_new.hdf5``. (This process takes the longest due to the size of the dataset.)
        """

        # download the original file if it is not in folder.
        fname_details = self.core.fname_bhs_details()
        if not os.path.exists(fname_details):
            self.download_details_file()

        # original data
        with h5py.File(fname_details, 'r') as f_dets:
            partIDs_details = f_dets['id'][:]
            time_details = f_dets['time'][:]

        # print(num_bad)

        change = self.change3

        for i in range(len(change)):
            # inds_fix = np.where((partIDs_details == change['id_remove'][i]) & (time_details >= change['time'][i]))[0]
            inds_fix = (partIDs_details == change['id_remove'][i]) & (time_details >= change['time'][i])

            if np.count_nonzero(inds_fix) == 0:
                print('No change needed')
            partIDs_details[inds_fix] = change['id_keep'][i]

            # Need to update change array.
            # inds_fix = np.where((change['id_keep'] == change['id_remove'][i]) & (change['time'] >= change['time'][i]))[0]
            inds_fix = (change['id_keep'] == change['id_remove'][i]) & (change['time'] >= change['time'][i])
            change['id_keep'][inds_fix] = change['id_keep'][i]

            # inds_fix = np.where((change['id_remove'] == change['id_remove'][i]) & (change['time'] >= change['time'][i]))[0]
            inds_fix = (change['id_remove'] == change['id_remove'][i]) & (change['time'] >= change['time'][i])
            change['id_remove'][inds_fix] = change['id_keep'][i]

            print(i)

        # read out
        with h5py.File(fname_details, 'a') as f_dets_new:
            f_dets_new.create_dataset('id_new', data=partIDs_details, dtype=partIDs_details.dtype.name, chunks=True, compression='gzip', compression_opts=9)

        return
