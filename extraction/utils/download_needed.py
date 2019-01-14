"""
MAIN PURPOSE: download all of the subhalos needed.
"""

import h5py
import numpy as np
import os
import shutil
import io

import tqdm

import illpy as ill

from utils.generalfuncs import download_sub
from utils import SubProcess

FLUSH_INT = 10


def write_complete(fio, row, flush=0):
    # number, which, snap, subhalo
    fio.write('%i\t%i\t%i\t%i\n' % (row[0], row[1], row[2], row[3]))
    if (flush+1) % FLUSH_INT == 0:
        fio.flush()

    flush = flush + 1
    return flush


class Download_Needed(SubProcess):
    """
    Download_Needed downloads all of the subhalos necessary for host galaxy characterization of the mergers. For host galaxies post-merger, particle information is downloaded for gas, star, bh, and dm particles. For host galaxies pre-merger, particle information is downloaded for star and bh particles.

        THIS CODE DOES NOT CHECK IF IT IS NEEDED. THE USER MUST KNOW STATUS USING ``completed_snaps_and_subs.txt`` COMPARED TO ``snaps_and_subs_needed.txt``.

        To use this file, you need to have a specific file structure within self.dir_output.
            '%i/%i_sub_cutouts/' % (snap, snap). The files are then stored in the struture as '%i/%i_sub_cutouts/cutout_%i_%i.hdf5' % (snap, snap, snap, sub).

            attributes:
                :param  ill_run - (int) - illustris run to use
                :param  dir_output - (str) - dir_output to work out of

    """

    def __init__(self, core, **kwargs):
        super().__init__(core)


    def download_needed_subhalos(self):
        """
        Download all the subhalos files needed.
        """
        fname_needed = self.core.fname_snaps_and_subs()
        snap_subs_needed = np.genfromtxt(fname_needed, skip_header=1, names=True, dtype=None)

        # need to keep track because this process will time out.
        fname_completed = self.core.fname_snaps_and_subs_completed()
        exists = os.path.exists(fname_completed)
        print("Completed snaps and subs: '{}' exists: {}".format(fname_completed, exists))

        if exists:
            f_complete = open(fname_completed, 'r+')
            lines_completed = f_complete.readlines()
            start = len(lines_completed)
            last = None if len(lines_completed) == 0 else lines_completed[-1]
            print("Start = '{}', last line: '{}'".format(start, last))
        else:
            f_complete = open(fname_completed, 'w')
            start = 0

        '''
        try:
            f_complete = open(fname_completed, 'r+')
        except FileNotFoundError:
            f_complete = open(fname_completed, 'w')

        # figure out where you left off
        try:
            lines_completed = f_complete.readlines()
            start = len(lines_completed)
            last = None if len(lines_completed) == 0 else lines_completed[-1]
            print("Start = '{}', last line: '{}'".format(start, last))
            print("snap_subs_needed[{}-1]: '{}'".format(start, snap_subs_needed[start-1]))
        except io.UnsupportedOperation:
            start = 0

        print(start)
        print(len(snap_subs_needed[start:]))
        '''

        # raise
        flush = 0
        for i, row in enumerate(tqdm.tqdm(snap_subs_needed[start:], desc='Snap subhalos needed')):
            which = row[1]
            snap = row[2]
            sub = row[3]
            fname_snap_sub_cutout = self.core.fname_snap_sub_cutout(snap, sub)
            '''
            if os.path.exists(fname_snap_sub_cutout):
                print("WARNING: i={}, row={}\n\tfile exists '{}'".format(
                    i, row, fname_snap_sub_cutout))
                flush = write_complete(f_complete, row, flush=flush)
                continue
            '''
            '''
            else:
                print(i, row)
                print("DOES NOT EXIST!")
                raise
            '''

            # Make sure output path exists
            path_snap_sub = os.path.dirname(fname_snap_sub_cutout)
            os.makedirs(path_snap_sub, exist_ok=True)

            # if which != 3, we only want stars and bhs
            # check if this file is already there
            if which != 3:
                if os.path.exists(fname_snap_sub_cutout):
                    with h5py.File(fname_snap_sub_cutout, 'r') as f:
                        if 'PartType4' in f:
                            # f_complete.write('%i\t%i\t%i\t%i\n' % (row[0], row[1], snap, sub))
                            flush = write_complete(f_complete, row, flush=flush)
                            continue

            # if which != 3, we only want stars and bhs
            if which != 3:
                cutout_request = {
                    'bhs': 'all',
                    'stars': 'Coordinates,Masses,Velocities,GFM_StellarFormationTime'
                }

            # if which == 3, get all particle data with quant ities of interest
            else:
                cutout_request = {
                    'bhs': 'all',
                    'gas': 'Coordinates,Masses,Velocities,StarFormationRate',
                    'stars': 'Coordinates,Masses,Velocities,GFM_StellarFormationTime',
                    'dm': 'Coordinates,Velocities'
                }

            # cutout = download_sub(self.base_url, snap, sub, cutout_request=cutout_request)
            cutout = self.get_cutout(snap, sub, cutout_request=cutout_request)

            # move file for good organization
            # os.rename(cutout, fname_snap_sub_cutout)
            shutil.move(cutout, fname_snap_sub_cutout)

            # record what has been completed
            # f_complete.write('%i\t%i\t%i\t%i\n' % (row[0], row[1], snap, sub))
            flush = write_complete(f_complete, row, flush=flush)

        return

    def get_cutout(self, snap, sub, cutout_request=None):
        return download_sub(self.base_url, snap, sub, cutout_request=cutout_request)


class Download_Needed_Odyssey(Download_Needed):
    """
    """

    '''
    def get_cutout(self, snap, sub, cutout_request=None):
        # return download_sub(self.base_url, snap, sub, cutout_request=cutout_request)

        import illpy.snapshot

        part_nums = [0, 1, 4, 5]
        # fields = None

        part_fields = {
            0: ['Coordinates', 'Masses', 'Velocities', 'StarFormationRate'],
            1: ['Coordinates', 'Velocities'],
            4: ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime'],
            5: None,
        }

        path = self.core.dir_input

        fname_out = "temp_snap{}_sub{}.hdf5".format(snap, sub)

        with h5py.File(fname_out, 'w') as out:
            for pt in part_nums:
                pt_key = "PartType{}".format(pt)
                fields = part_fields[pt]
                group = out.create_group(pt_key)
                subh = ill.snapshot.loadSubhalo(path, snap, sub, pt, fields=fields)
                keys = list(subh.keys())
                # print("{} : keys = '{}'".format(pt, keys))
                for key in keys:
                    group[key] = subh[key]

        # print("File '{}' size: {}".format(fname_out, os.path.getsize(fname_out)))

        # raise
        return fname_out
    '''

    def get_cutout(self, snap, sub, cutout_request=None):
        # return download_sub(self.base_url, snap, sub, cutout_request=cutout_request)

        import illpy.snapshot

        part_nums = [0, 1, 4, 5]
        # fields = None

        '''
        part_fields = {
            0: ['Coordinates', 'Masses', 'Velocities', 'StarFormationRate'],
            1: ['Coordinates', 'Velocities'],
            4: ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime'],
            5: None,
        }
        '''

        # Determine what parameters/fields we want from the cutout (subhalo)
        if cutout_request is None:
            part_fields = {
                0: ['Coordinates', 'Masses', 'Velocities', 'StarFormationRate'],
                1: ['Coordinates', 'Velocities'],
                4: ['Coordinates', 'Masses', 'Velocities', 'GFM_StellarFormationTime'],
                5: None,
            }
        else:
            part_fields = {}
            for key, val in cutout_request.items():
                pnum = ill.snapshot.partTypeNum(key)
                fields = [vv.strip() for vv in val.strip().split(',')]
                fields = None if 'all' in fields else fields
                part_fields[pnum] = fields

        path = self.core.dir_input
        fname_out = "temp_snap{}_sub{}.hdf5".format(snap, sub)

        with h5py.File(fname_out, 'w') as out:
            out_keys = list(out.keys())
            for pt, fields in part_fields.items():
                pt_key = "PartType{}".format(pt)
                # fields = part_fields[pt]
                group = out.create_group(pt_key)
                subh = ill.snapshot.loadSubhalo(path, snap, sub, pt, fields=fields)
                keys = list(subh.keys())
                # print("{} : keys = '{}'".format(pt, keys))
                for key in keys:
                    group[key] = subh[key]

        # print("File '{}' size: {}".format(fname_out, os.path.getsize(fname_out)))

        # raise
        return fname_out
