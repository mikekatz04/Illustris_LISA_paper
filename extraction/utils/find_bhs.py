"""
MAIN PURPOSE: locate all the black holes and match them to their subhalos
"""

import h5py
import numpy as np
import snapshot
import os

from utils.generalfuncs import get

h = 0.704


class LocateBHs:
    """
    This class locates all of the black holes through the simulation as each snapshot. To do this, it downloads all the bh particle information from the snapshot chunks. It also downloads group catalog files for header (offset) information. It uses the snapshot python function provided in the illustris python scripts to locate which subhalos have each black hole particle. It then reads out the data to a file. This process is done separately for each snapshot in case downloading times out.

        After this process is completed, all of the bh files for each snapshot are combined into an output file ``bhs_all_new.hdf5``.

        attributes:

            :param  ill_run - (int) - which illustris run
            :param  dir_output - (string) - base dir_output
            :param  num_chunk_files_per_snapshot - (int) - number of chunk files per snapshot
            :param  num_groupcat_files - (int) - number of group files needed at each snapshot for offset information
            :param  first_snap_with_bhs - (int) - first snapshot where black holes exist
            :param  skip_snaps - list of (int) - snaps missing in specific illustris run
            :param  max_snap - (int) - highest snap in simulation

            base_url - (str) - base url for downloading files
            snaps - array of (int) - snapshot numbers of all subhalos with black holes in them
            subs - array of (int) - subhalo nubers of all subhalos with black holes in them
            needed - (boolean) - if this code needs to run
            bhs_dict - dict - all of the black hole particle information categories contained in simulation

        methods:

            reset_black_holes_dict
            download_bhs_all_snapshots
            download_bhs_for_snapshot
            download_snapshot_files
            download_groupcat_header_file
            populate_bh_dict
            read_out_to_file_snapshot
            delete_snapshot_files
            combine_black_hole_files
            delet_snap_bh_files
    """

    def __init__(self, ill_run=1, dir_output='./extraction_files/', num_chunk_files_per_snapshot=512, num_groupcat_files=1, first_snap_with_bhs=30, skip_snaps=[53, 55], max_snap=135):

        self.dir_output = dir_output
        self.num_chunk_files_per_snapshot = num_chunk_files_per_snapshot
        self.num_groupcat_files = num_groupcat_files
        self.first_snap_with_bhs = first_snap_with_bhs
        self.max_snap = max_snap
        self.skip_snaps = skip_snaps

        self.base_url = "http://www.illustris-project.org/api/Illustris-%i/" % ill_run

        # load which subs have black holes
        with h5py.File('subs_with_bhs.hdf5', 'r') as f:
            self.snaps = f['Snapshot'][:]
            self.subs = f['SubhaloID'][:]

        # initialize the black hole dict
        self.reset_black_holes_dict()

        # check if this process is needed
        if 'bhs_all_new.hdf5' in os.listdir(self.dir_output):
            self.needed = False

        else:
            self.needed = True

    def reset_black_holes_dict(self):
        """
        The black holes dict carries all information about the black holes, including the values for each quantity, the names all qunatities, the conversion factors to familiar units, and the datatype for readout.
        """

        self.bhs_dict = {
            'BH_CumEgyInjection_QM': {'unit': r'$M_odot/(ckpc^2\ Gyr^2)$', 'dtype': 'f', 'cf': 1e10/(0.978**2*h), 'values': []},
            'BH_CumMassGrowth_QM': {'unit': r'$M_\odot$', 'dtype': 'f', 'cf': 1e10/h, 'values': []},
            'BH_Density': {'unit': r'$M_\odot/ckpc^3$', 'dtype': 'f', 'cf': 1e10*h**2, 'values': []},
            'BH_Hsml': {'unit': 'ckpc', 'dtype': 'f', 'cf': 1/h, 'values': []},
            'BH_Mass': {'unit': r'$M_\odot$', 'dtype': 'f', 'cf': 1e10/h, 'values': []},
            'BH_Mass_bubbles': {'unit': r'$M_\odot$', 'dtype': 'f', 'cf': 1e10/h, 'values': []},
            'BH_Mass_ini': {'unit': r'$M_\odot$', 'dtype': 'f', 'cf': 1e10/h, 'values': []},
            'BH_Mdot': {'unit': r'$M_\odot$/year', 'dtype': 'f', 'cf': 10.22, 'values': []},
            'BH_Pressure': {'unit': r'$M_odot/(ckpc\ Gyr^2)$', 'dtype': 'f', 'cf': 1e10*h**3/(0.978**2), 'values': []},
            'BH_Progs': {'unit': 'None', 'dtype': 'i', 'cf': 'None', 'values': []},
            'BH_U': {'unit': r'$(km/s)^2$', 'dtype': 'f', 'cf': 'None', 'values': []},
            'Coordinates': {'unit': 'ckpc', 'dtype': 'f', 'cf': 1/h, 'values': []},
            'HostHaloMass': {'unit': r'$M_\odot$', 'dtype': 'f', 'cf': 1e10/h, 'values': []},
            'Masses': {'unit': r'$M_\odot$', 'dtype': 'f', 'cf': 1e10/h, 'values': []},
            'NumTracers': {'unit': 'None', 'dtype': 'i', 'cf': 'None', 'values': []},
            'ParticleIDs': {'unit': 'None', 'dtype': 'long', 'cf': 'None', 'values': []},
            'Potential': {'unit': r'$(km/s)^2/a$', 'dtype': 'f', 'cf': 'None', 'values': []},
            'SubfindDensity': {'unit': r'$M_odot/ckpc^3$', 'dtype': 'f', 'cf': 1e10*h**2, 'values': []},
            'SubfindHsml': {'unit': 'ckpc', 'dtype': 'f', 'cf': 1/h, 'values': []},
            'SubfindVelDisp': {'unit': 'km/s', 'dtype': 'f', 'cf': 'None', 'values': []},
            'Velocities': {'unit': r'$km\sqrt(a)/s$', 'dtype': 'f', 'cf': 'None', 'values': []},
            'Snapshot': {'unit': 'None', 'dtype': 'i', 'cf': 'None', 'values': []},
            'Subhalo': {'unit': 'None', 'dtype': 'i', 'cf': 'None', 'values': []}
        }

        return

    def download_bhs_all_snapshots(self):
        """
        This downloads all the black holes per snapshot.
        """

        for snap in np.arange(self.first_snap_with_bhs, self.max_snap+1):
            if snap in self.skip_snaps:
                print('Skipped snapshot', snap)

            if '%i_blackholes.hdf5' % snap in os.listdir(self.dir_output):
                continue

            self.download_bhs_for_snapshot(snap)
            self.reset_black_holes_dict()

    def download_bhs_for_snapshot(self, snap):
        """
        This downloads black holes for a specific snapshot. First, the snapshot and group catalog chunk files are needed. To do this a specific file structure is needed. See http://www.illustris-project.org/data/docs/scripts/ for info.
        """

        if '%i/' % snap not in os.listdir(self.dir_output):
            os.mkdir(self.dir_output + '%i/' % snap)

        print('Begin download snapshot file black hole info for snapshot', snap)
        self.download_snapshot_files(snap)
        print('Finished downloading snapshot file black hole info for snapshot', snap)

        print('Begin download groupcat file for snapshot', snap)
        self.download_groupcat_header_file(snap)
        print('Finished downloading groupcat file for snapshot', snap)

        self.populate_bh_dict(snap)
        self.read_out_to_file_snapshot(snap)

        self.delete_snapshot_files(snap)

        print('Finished gathering snapshot particle data for bhs for snapshot', snap)
        return

    def download_snapshot_files(self, snap):
        """
        Download all of the chunk files for the current snapshot. Special file structure is needed. This only downloads the black hole information for memory conservation.
        """

        if 'snapdir_%03d/' % snap not in os.listdir(self.dir_output + '%i/' % snap):
            os.mkdir(self.dir_output + '%i/' % snap + 'snapdir_%03d/' % snap)

        for chunk_num in range(self.num_chunk_files_per_snapshot):
            if 'snap_%i.%i.hdf5' % (snap, chunk_num) in os.listdir(self.dir_output + '%i/' % snap + 'snapdir_%03d/' % snap):
                continue
            cutout = get(self.base_url + "files/snapshot-" + str(snap) + '.'  + str(chunk_num) + '.hdf5?bhs=all')
            os.rename(cutout, self.dir_output + '%i/' % snap + 'snapdir_%03d/' % snap + cutout)
            if chunk_num % 10 == 0:
                print('Snapshot chunk', chunk_num, 'out of', self.num_chunk_files_per_snapshot, 'completed.')
        return

    def download_groupcat_header_file(self, snap):
        """
        Download the group catalog files which act as header files for `snapshot.py` (http://www.illustris-project.org/data/docs/scripts/ for info on this script.) Usually the first group catalog file is all that is needed for black hole particle informationself.
        """

        if 'groups_%03d/' % snap not in os.listdir(self.dir_output + '%i/' % snap):
            os.mkdir(self.dir_output + '%i/' % snap + 'groups_%03d/' % snap)

        for chunk_num in range(self.num_groupcat_files):
            if 'groups_%i.%i.hdf5' % (snap, chunk_num) in os.listdir(self.dir_output + '%i/' % snap + 'snapdir_%03d/' % snap):
                continue
            cutout = get(self.base_url + "files/groupcat-" + str(snap) + '.'  + str(chunk_num) + '.hdf5')
            os.rename(cutout, self.dir_output + '%i/' % snap + 'groups_%03d/' % snap + cutout)
            if chunk_num % 1 == 0:
                print('Groupcat chunk', chunk_num, 'out of', self.num_chunk_files_per_snapshot, 'completed.')

        return

    def populate_bh_dict(self, snap):
        """
        Use `snapshot.py` to find all the black hole particles in each subhalo that has a black hole in it. You need the snapshot particle chunks and group catalog header information. This tells you the black holes in each subhalo which is important for pairing a black hole to its host galaxy. These values are populated in the bh_dict.
        """

        # figure out which subs are in this specific snapshot
        subs_to_look_in = self.subs[self.snaps == snap]
        for sub in subs_to_look_in:

            # load bh particle info from the snapshot folder using `snapshot.py`
            check_bhs_in_sub = snapshot.loadSubhalo(self.dir_output + '%i/' % snap, snap, sub, 5, fields=None)

            # need length to fill values for subhalo and snapshot
            length = len(check_bhs_in_sub['ParticleIDs'][:])
            self.bhs_dict['Subhalo']['values'].append(np.full((length, ), sub, dtype=int))
            self.bhs_dict['Snapshot']['values'].append(np.full((length, ), snap, dtype=int))

            for name in self.bhs_dict:
                if name == 'Subhalo' or name == 'Snapshot':
                    continue
                self.bhs_dict[name]['values'].append(check_bhs_in_sub[name])

        return

    def read_out_to_file_snapshot(self, snap):
        """
        Concatenate the information in each list of the bh dict and then read them out to a file specific to the bhs in this snapshot.
        """
        with h5py.File(self.dir_output + '%i/%i_blackholes.hdf5' % (snap, snap), 'w') as f:
            for name in self.bhs_dict:
                output = np.concatenate(self.bhs_dict[name]['values'], axis=0)

                # multiply by a conversion factor if there is one
                if self.bhs_dict[name]['cf'] != 'None':
                    output = output * self.bhs_dict[name]['cf']

                dset = f.create_dataset(name, data=output, dtype=output.dtype.name, chunks=True, compression='gzip', compression_opts=9)

                dset.attrs['unit'] = self.bhs_dict[name]['unit']

        return

    def delete_snapshot_files(self, snap):
        """
        Delete the snapshot chunks and group catalog chunks (header files). First check to make sure the output file is there.
        """

        # make sure black hole file is there!!!
        if '%i_blackholes.hdf5' % (snap) not in os.listdir(self.dir_output + '%i/' % snap):
            raise Exception('About to delete files when completed file (%i_blackholes.hdf5) is not there.' % snap)

        for f in os.listdir(self.dir_output + '%i/' % snap + 'snapdir_%03d/' % snap):
            os.remove(self.dir_output + '%i/' % snap + 'snapdir_%03d/' % snap + f)

        for f in os.listdir(self.dir_output + '%i/' % snap + 'groups_%03d/' % snap):
            os.remove(self.dir_output + '%i/' % snap + 'groups_%03d/' % snap + f)

        return

    def combine_black_hole_files(self):
        """
        Combine all the ``(snap)_blackholes.hdf5`` into a single file: ``bhs_all_new.hdf5``.
        """

        # reset the dict so it is ready to populate
        self.reset_black_holes_dict()
        bhs_dict = self.bhs_dict

        # open snapshot specific files and populate dict
        for snap in np.arange(self.first_snap_with_bhs, self.max_snap+1):
            with h5py.File(self.dir_output + '%i/%i_blackholes.hdf5' % (snap, snap), 'r') as f:
                for name in bhs_dict:
                    self.bhs_dict[name]['values'].append(f[name][:])

        # concatenate each list
        for name in self.bhs_dict:
            output = np.concatenate(self.bhs_dict[name]['values'], axis=0)
        # to be certain the items are ordered properly, place in a structured array
        # and sort by snapshot and then subhalo.
        checker = np.array([(bhs_dict['Snapshot']['values'][i], bhs_dict['Subhalo']['values'][i]) for i in range(len(bhs_dict['Subhalo']['values']))], dtype=[('Snapshot', np.dtype(np.uint64)), ('Subhalo', np.dtype(np.uint64))])

        sort = np.argsort(checker, order=('Snapshot', 'Subhalo'))

        print('Write out to combined file.')
        with h5py.File('bhs_all_new.hdf5', 'w') as f:
            for name in bhs_dict:
                output = bhs_dict[name]['values'][sort]
                dset = f.create_dataset(name, data=output, dtype=output.dtype.name, chunks=True, compression='gzip', compression_opts=9)
                dset.attrs['unit'] = bhs_dict[name]['unit']

        self.delete_snap_bh_files()
        return

    def delet_snap_bh_files(self):
        """
        Delete the snapshot specific bh files to conserve memory.
        """

        for snap in np.arange(self.first_snap_with_bhs, self.max_snap+1):
            os.remove(self.dir_output + '%i/%i_blackholes.hdf5' % (snap, snap))

        return
