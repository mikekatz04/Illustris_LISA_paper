"""
MAIN PURPOSE: determine which mergers are good and bad.

    test_good_bad_mergers.py is used to determine if the merger is good or bad in terms of its black holes. The criteria to be a bad merger is the following:

        1) Either constituent black hole respawned after a fly-by encounter of its host galaxy lost the host galaxy's original black hole. This is caught by FindBadBlackHoles class and recorded in ``bad_black_holes.txt``. (This applies to all black holes regardless of mass. However, these are expected to be near seed mass.)
        2) Either constituent exists in the simulation for 1 or less snapshots. This only applies to black holes less than 10^6. The algorithms used here are not perfect and can let a few of these slip by. Therefore, if a black hole is larger than 10^6, it is considered good no matter how long its ParticleID has existed. (We assume if it is larger than 10^6, it has been around longer than 2 snapshots.)
"""

import os
import h5py
import numpy as np
import tqdm

from utils import SubProcess


class FindBadBlackHoles(SubProcess):
    """
    FindBadBlackHoles walks down trees in the sublink trees locating bad black holes. This is the key aspect to the more in-depth analysis to access the near-seed mass black holes. What we mean by bad black holes is the following: when a subhalo loses its black hole in a fly-by encounter and subsequently spawns a new black hole, the spawned black hole is considered bad.

        List of the steps we take to do this (this is after we have filtered for black holes that stop existing at snapshots earlier than the last snapshot):

        step 1: start at the subhalo where a black hole is about to cease to exist. Its first descendant will either be the galaxy it has merged into, or it will remain on its own except without a black hole.

        step 2: iterate down the descendant tree until the descendant index is -1

        (as we iterate down the tree)

        step 3: find if a subhalo we are looking at has a black hole

        step 4: if it has a black hole, check if this black hole's first snapshot (i.e. its formation snapshot) is the same as the snapshot of the current descendant in the tree.

            - If True: this means the black hole formed into the subhhalo that lost its black hole (making it a bad black hole).

            - If False: it is not he same as the formation snapshot, that means the subhalo we are looking at merged into another subhalo with a black hole that had existed at previous snapshots.

        step 5: add bad black holes to list and read out the data

        attributes:
            :param  dir_output - (str) - dir_output to work in

            needed - (bool) - if this code needs to run

        methods:
            get_subs_from_all_bhs
            search_bad_black_holes


        MAIN JOB: locate bad black holes resulting from fly-by encounters of host galaxies
    """
    def __init__(self, main_proc):
        super().__init__(main_proc)

        fname_bads = self.core.fname_bad_mergers()
        if os.path.exists(fname_bads):
            self.needed = False
        else:
            self.needed = True

    def search_bad_black_holes(self):
        """
        Find bad black holes using the method described in the FindBadBlackHoles class description.
        """

        # get all the information from the all black holes catalog needed
        subID_raw_filtered, part_ids_new, part_ids_old, snaps_all, subID_raw_all, bh_masses_all = \
            self.get_subs_from_all_bhs()

        # open the sublink tree file. Keep it open because it is
        # too much information to store it all in memory
        fname_sublink = self.core.fname_sublink_short()
        # f_sublink = h5py.File(fname_sublink, 'r')
        with h5py.File(fname_sublink, 'r') as f_sublink:

            # get the raw IDs from sublink and sort them
            subID_raw_sublink = np.asarray(
                f_sublink['SubfindID'][:310600757] + f_sublink['SnapNum'][:310600757]*1e12,
                dtype=np.int64)

            inds_sort = np.argsort(subID_raw_sublink)

            # find the raw sub IDs for subhalos at the last snapshot the black holes exist.
            # We can do this by index because of our initial work with adding
            # decendant and subhaloID indices in addition to the real values.
            final_subs_inds = inds_sort[
                np.searchsorted(subID_raw_sublink[inds_sort], subID_raw_filtered)
            ]

            bad = []
            # print(len(final_subs_inds), 'to look at')
            for j, final_ind in enumerate(tqdm.tqdm(final_subs_inds, desc='Subhalo inds')):
                desc_ind = final_ind

                # iterate down the tree
                while desc_ind != -1:

                    # using indices makes this easy
                    #    play with these files separately if you uncertain of this procedure
                    # next descendant
                    desc_ind = f_sublink['Descendant_index'][desc_ind]

                    # check if the descendant will have a black hole in it
                    if f_sublink['SubhaloLenType'][desc_ind][5] > 0:

                        # figure out which subhalo in all bhs catalog corresponds to this sub
                        #    with a black hole from the sublink catalog
                        ind_sub = np.where(subID_raw_all == subID_raw_sublink[desc_ind])[0][0]
                        # ind_sub = (subID_raw_all == subID_raw_sublink[desc_ind])

                        # find all the snapshots that contain the black hole
                        #    that is in this subhalo of interest
                        # snaps_bh = snaps_all[np.where(part_ids_new == part_ids_new[ind_sub])]
                        snaps_bh = snaps_all[(part_ids_new == part_ids_new[ind_sub])]

                        # check if the first snapshot this black hole exists
                        #    is the same as the descendant
                        if snaps_bh[0] == f_sublink['SnapNum'][desc_ind]:

                            # append to bad list
                            bad.append((part_ids_new[ind_sub],
                                        bh_masses_all[ind_sub],
                                        f_sublink['SnapNum'][desc_ind]))

            # read out
            dtype = [
                ('id', np.dtype(np.uint64)),
                ('mass', np.dtype(float)),
                ('snap', np.dtype(np.int32))
            ]
            bad_arr = np.asarray(bad, dtype=dtype)

            fname_bads = self.core.fname_bad_mergers()
            np.savetxt(fname_bads, bad_arr)

        # f_sublink.close()

        return

    def get_subs_from_all_bhs(self):
        """
        Get information from the all bhs catalog to find the information for the last snapshot black holes exist. This information is also needed as we walk down the sublink tree.
        """

        # get initial information and calculate raw subhalo id number according to Illustris convention
        fname_all = self.core.fname_bhs_all()
        with h5py.File(fname_all, 'r') as f_all_bhs:
            part_ids_new = f_all_bhs['ParticleIDs_new'][:]
            part_ids_old = f_all_bhs['ParticleIDs'][:]
            snaps_all = f_all_bhs['Snapshot'][:]
            subhalos_all = f_all_bhs['Subhalo'][:]
            subID_raw = np.asarray(snaps_all*1e12 + subhalos_all, dtype=np.int64)
            bh_masses = f_all_bhs['BH_Mass'][:]

        # find the last time you find a unique id
        ids_unique, index = np.unique(part_ids_new[::-1], return_index=True)

        # only need to look at black holes where the last time
        # we see them are before snapshot 135
        # filter_inds = np.where(snaps_all[::-1][index] != 135)[0]
        filter_inds = (snaps_all[::-1][index] != 135)

        # this is the filtered version of subID_raw
        subID_raw_filtered = np.asarray(
            snaps_all[::-1][index][filter_inds]*1e12 + subhalos_all[::-1][index][filter_inds],
            dtype=np.int64)

        return subID_raw_filtered, part_ids_new, part_ids_old, snaps_all, subID_raw, bh_masses


class TestGoodBadMergers(SubProcess):
    """
        TestGoodBadMergers tests whether either constituent exists in the simulation for 1 or less snapshots. This only applies to black holes less than 10^6. The algorithms used here are not perfect and can let a few of these slip by. Therefore, if a black hole is larger than 10^6, it is considered good no matter how long its ParticleID has existed. (We assume if it is larger than 10^6, it has been around longer than 2 snapshots.)

        attributes:
            :param  dir_output - (str) - dir_output to work in

            needed - (bool) - if this code needs to run

        methods:
            get_subs_from_all_bhs
            search_bad_black_holes
    """

    def __init__(self, core):
        super().__init__(core)

        fname_goods = self.core.fname_good_mergers()
        if os.path.exists(fname_goods):
            self.needed = False
        else:
            self.needed = True

    def test_mergers(self):
        """
        Test if the mergers are good or bad. Criterion is stated in the description of the TestGoodBadMergers class object.
        """

        fname_all = self.core.fname_bhs_all()
        with h5py.File(fname_all, 'r') as f:

            # get the unique ides in the all bhs dataset, as well as their first appearence (index) and count of appearences (counts)
            unique_part_ids_new, index, counts = np.unique(f['ParticleIDs_new'][:][::-1], return_counts=True, return_index=True)

            # REMOVING ALL BLACK HOLES THAT ONLY APPEAR FOR ONE SNAPSHOT IN THE FULL BH-ALL DATASET and that have mass less than 10^6. (The merger switch algoritm is not perfect, so we cut out any issues below 10^6) #
            bad_add = np.where((counts == 1) & (f['BH_Mass'][:][::-1][index] > 1e6))[0]

        # read in bad black holes
        fname_bads = self.core.fname_bad_mergers()
        bad_arr = np.genfromtxt(fname_bads, dtype=None)

        bad_ids = bad_arr[:, 0].astype(np.uint64)

        # combine the black holes from the fly-by/descendant search with black holes that exist less than 1 snapshot (less than 10^6)
        bad_ids = np.unique(np.concatenate([bad_ids, unique_part_ids_new[bad_add]]))

        # GOOD MERGER IF THE BOTH CONSTITUENT BLACKHOLES APPEAR MORE THAN ONE TIME IN THE ''ALL'' DATASET, AND ARE NOT CAUGHT IN `find_bad_black_holes.py` to be rebirthed #
        good = []

        # read in merger data
        fname_mergers = self.core.fname_bhs_mergers()
        with h5py.File(fname_mergers, 'r') as f_merg:
            time = f_merg['time'][:]
            mass_in_new = f_merg['mass_in_new'][:]
            mass_out_new = f_merg['mass_out_new'][:]
            id_in_new = f_merg['id_in_new'][:]
            id_out_new = f_merg['id_out_new'][:]

        print('find good/bad')
        for m in range(len(time)):

            # if both masses are above 10^6, we keep it
            if mass_in_new[m] >= 1e6 and mass_out_new[m] >= 1e6:
                good.append(m)
                continue

            # check if either constituent is in the bad_ids
            if id_in_new[m] not in bad_ids and id_out_new[m] not in bad_ids:

                # make sure both constituent bhs are in the particle IDs in the all bhs dataset
                if id_in_new[m] in unique_part_ids_new and id_out_new[m] in unique_part_ids_new:
                    good.append(m)

        # read out
        fname_goods = self.core.fname_good_mergers()
        np.savetxt(fname_goods, np.array([np.array(good)]).T)
        return
