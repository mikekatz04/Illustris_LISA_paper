"""
MAIN PURPOSE: download all of the subhalos needed.
"""

import h5py
import numpy as np
import os
import io

import tqdm

from utils.generalfuncs import download_sub
from utils import SubProcess


class DownloadNeeded(SubProcess):
    """
    DownloadNeeded downloads all of the subhalos necessary for host galaxy characterization of the mergers. For host galaxies post-merger, particle information is downloaded for gas, star, bh, and dm particles. For host galaxies pre-merger, particle information is downloaded for star and bh particles.

        THIS CODE DOES NOT CHECK IF IT IS NEEDED. THE USER MUST KNOW STATUS USING ``completed_snaps_and_subs.txt`` COMPARED TO ``snaps_and_subs_needed.txt``.

        To use this file, you need to have a specific file structure within self.dir_output.
            '%i/%i_sub_cutouts/' % (snap, snap). The files are then stored in the struture as '%i/%i_sub_cutouts/cutout_%i_%i.hdf5' % (snap, snap, snap, sub).

            attributes:
                :param  ill_run - (int) - illustris run to use
                :param  dir_output - (str) - dir_output to work out of

    """

    def __init__(self, core, **kwargs):
        # self.ill_run = ill_run
        # self.dir_output = dir_output
        super().__init__(core)

        # NO CHECK IF IT IS NEEDED

    def download_needed_subhalos(self):
        """
        Download all the subhalos files needed.
        """
        # base_url = "http://www.illustris-project.org/api/Illustris-%i/" % self.ill_run

        # fname = os.path.join(self.dir_output, 'snaps_and_subs_needed.txt')
        fname_needed = self.core.fname_snaps_and_subs()
        snap_subs_needed = np.genfromtxt(fname_needed, skip_header=1, names=True, dtype=None)

        # need to keep track because this process will time out.
        fname_completed = self.core.fname_snaps_and_subs_completed()
        try:
            f_complete = open(fname_completed, 'r+')
        except FileNotFoundError:
            f_complete = open(fname_completed, 'w')

        # figure out where you left off
        try:
            lines_completed = f_complete.readlines()
            start = len(lines_completed)
            print("Start = '{}', last line: '{}'".format(start, lines_completed[-1]))
            print("snap_subs_needed[{}-1]: '{}'".format(start, snap_subs_needed[start-1]))
        except io.UnsupportedOperation:
            start = 0

        print(start)
        print(len(snap_subs_needed[start:]))
        for i, row in enumerate(tqdm.tqdm(snap_subs_needed[start:], desc='Snap subhalos needed')):
            which = row[1]
            snap = row[2]
            sub = row[3]
            fname_snap_sub_cutout = self.core.fname_snap_sub_cutout(snap, sub)
            # Make sure output path exists
            path_snap_sub = os.path.dirname(fname_snap_sub_cutout)
            os.makedirs(path_snap_sub, exist_ok=True)

            # if which != 3, we only want stars and bhs
            # check if this file is already there
            if which != 3:
                # if 'cutout_%i_%i.hdf5' % (snap, sub) in os.listdir('%i/%i_sub_cutouts/' % (snap, snap)):
                if os.path.exists(fname_snap_sub_cutout):
                    with h5py.File(fname_snap_sub_cutout, 'r') as f:
                        if 'PartType4' in f:
                            f_complete.write('%i\t%i\t%i\t%i\n' % (row[0], row[1], snap, sub))
                            print('already there')
                            continue

            # if which != 3, we only want stars and bhs
            if which != 3:
                cutout_request = {'bhs': 'all', 'stars': 'Coordinates,Masses,Velocities,GFM_StellarFormationTime'}

            # if which == 3, get all particle data with quant ities of interest
            else:
                cutout_request = {'bhs': 'all', 'gas': 'Coordinates,Masses,Velocities,StarFormationRate', 'stars': 'Coordinates,Masses,Velocities,GFM_StellarFormationTime', 'dm': 'Coordinates'}

            cutout = download_sub(self.base_url, snap, sub, cutout_request=cutout_request)

            # if '%i' % snap not in os.listdir(self.dir_output):
            #     os.mkdir('%i/' % snap)
            #
            # if '%i_sub_cutouts' % snap not in os.listdir(self.dir_output + '%i/' % snap):
            #     os.mkdir(self.dir_output + '%i/' % snap + '%i_sub_cutouts/' % snap)

            # move file for good organization
            os.rename(cutout, fname_snap_sub_cutout)

            # for testing in local folder
            # os.rename(cutout, 'cutout_%i_%i.hdf5' % (snap, sub))

            # record what has been completed
            f_complete.write('%i\t%i\t%i\t%i\n' % (row[0], row[1], snap, sub))

            # if (i+1) % 1 == 0:
            #     print(i + start + 1)

        return
