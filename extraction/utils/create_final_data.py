"""
MAIN PURPOSE: create the final dataset of information after all of these analyses
"""

import os
import sys

import numpy as np
import h5py
import tqdm

from utils import SubProcess

h = 0.704
DEBUG = True


class Create_Final_Data(SubProcess):
    """
    CreateFinalDataset gathers all of the data from the entire analysis and combines it into a single output file. This includes only the good mergers.

        attributes:
            :param  dir_output - (str) - dir_output to work in
            needed - (bool) - if this code needs to run

        methods:
            gather_info_from_all_bhs
            gather_info_from_mergers
            gather_info_from_subs_with_bhs
            gather_density_profiles_and_vel_disps
            create_final_data
    """

    def __init__(self, core):
        super().__init__(core)

        fname = core.fname_final_data()
        self.needed = self._check_needed(fname)

        return

    def gather_info_from_all_bhs(self):
        """
        Get info for bhs from all bhs catalog.
        """

        fname = self.core.fname_bhs_all()
        with h5py.File(fname, 'r') as bh_all:
            if self.core.DEBUG:
                print("bhs_all ({})\n\tkeys: {}".format(fname, list(bh_all.keys())))
            bh_all_partids = bh_all['ParticleIDs_new'][:]
            bh_all_snapshots = bh_all['Snapshot'][:]
            bh_all_hsml = bh_all['BH_Hsml'][:]
            bh_all_coordinates = bh_all['Coordinates'][:]
            bh_all_mdot = bh_all['BH_Mdot'][:]

        return bh_all_partids, bh_all_snapshots, bh_all_hsml, bh_all_coordinates, bh_all_mdot

    def gather_info_from_mergers(self, uni_mergers):
        """
        Get info on mergers.
        """

        fname = self.core.fname_bhs_mergers()
        with h5py.File(fname, 'r') as mergers:
            if self.core.DEBUG:
                print("mergers ({})\n\tkeys: {}".format(fname, list(mergers.keys())))

            mass_in = mergers['mass_in_new'][:][uni_mergers]*1e10/h
            mass_out = mergers['mass_out_new'][:][uni_mergers]*1e10/h

            id_in_old = mergers['id_in'][:][uni_mergers]
            id_out_old = mergers['id_out'][:][uni_mergers]
            id_in_new = mergers['id_in_new'][:][uni_mergers]
            id_out_new = mergers['id_out_new'][:][uni_mergers]

            scale = mergers['time'][:][uni_mergers]
            redshift = 1./scale - 1.
            snapshots = mergers['snapshot'][:][uni_mergers]

        return mass_in, mass_out, id_in_old, id_out_old, id_in_new, id_out_new, scale, redshift, snapshots

    def gather_info_from_subs_with_bhs(self):
        """
        Get info for subs that have the bhs.
        """

        fname = self.core.fname_subs_with_bhs()
        with h5py.File(fname, 'r') as gc:
            if self.core.DEBUG:
                print("groupcat ({})\n\tkeys: {}".format(fname, list(gc.keys())))
            gc_subs = gc['SubhaloID'][:]
            gc_snaps = gc['Snapshot'][:]
            gc_SubhaloMassType = gc['SubhaloMassType'][:]
            gc_SubhaloVelDisp = gc['SubhaloVelDisp'][:]

        return gc_subs, gc_snaps, gc_SubhaloMassType, gc_SubhaloVelDisp

    def gather_density_profiles_and_vel_disps(self):
        """
        Get stellar velocity dispersions and density profiles.
        """

        fname_vdisp = self.core.fname_vel_disp()
        vel_disp = np.genfromtxt(fname_vdisp, names=True, dtype=None)
        if self.core.DEBUG:
            print("vel_disp ({})\n\tkeys: {}".format(fname_vdisp, vel_disp.dtype.names))

        # use the density profiles dataset to determine the final good mergers
        fname_dens = self.core.fname_density_profiles()
        dens_prof = np.genfromtxt(fname_dens, names=True, dtype=None)
        if self.core.DEBUG:
            print("dens_prof ({})\n\tkeys: {}".format(fname_vdisp, dens_prof.dtype.names))

        return vel_disp, dens_prof

    def create_final_data(self):
        """
        Gather all data and write it out to a dataset.
        """

        velocity_dispersions, dens_profs = self.gather_density_profiles_and_vel_disps()

        # all the unique mergers appearing in the density profiles
        uni_mergers = np.unique(dens_profs['m'])

        # unique mergers in velocity dispersions which is the same as in density profiles.
        # Also, get the indices of each unique entry
        uni_vel, uni_vel_ind = np.unique(velocity_dispersions['m'], return_index=True)

        bh_all_partids, bh_all_snapshots, bh_all_hsml, bh_all_coordinates, bh_all_mdot = \
            self.gather_info_from_all_bhs()
        mass_in, mass_out, id_in_old, id_out_old, id_in_new, id_out_new, scale, redshift, snapshots = \
            self.gather_info_from_mergers(uni_mergers)
        gc_subs, gc_snaps, gc_SubhaloMassType, gc_SubhaloVelDisp = \
            self.gather_info_from_subs_with_bhs()

        # initialize lists for all the parameters we want to read out
        subs_out = []
        snaps_out = []
        vel_disps_out = []
        vel_disps_from_subs_out = []
        stellar_mass_out = []
        total_mass_out = []
        separation_out = []
        mdot_out = []
        merger_ind_out = []
        ids_new_out = []
        ids_old_out = []
        masses_out = []
        density_profiles_out = []
        redshift_out = []
        coordinates_out = []

        # make a cut based on snapshots lived by black hole.
        # We already made this cut in previous analysis. This is here in case the user
        # wants a longer cut or to ensure this cut is correct.
        snapshot_cut = 1

        # run through the mergers and gather all the output information
        for j in tqdm.trange(len(uni_mergers), desc='Mergers'):

            # merger specifics
            idn_i = id_in_new[j]
            idn_o = id_out_new[j]
            ido_i = id_in_old[j]
            ido_o = id_out_old[j]

            msi = mass_in[j]
            mso = mass_out[j]

            snap = snapshots[j]

            # inds_in = np.where((bh_all_partids == idn_i) & (bh_all_snapshots < snap))[0]
            inds_in = (bh_all_partids == idn_i) & (bh_all_snapshots < snap)

            # recheck snapshot cut
            if (np.count_nonzero(inds_in) < snapshot_cut) and (msi < 1e6):
                print("Warning: found unexpected cut at `uni_merger` j={}".format(j))
                continue

            # get the separation for this black hole
            # bh_sep_in = bh_all_hsml[inds_in[-1]]*scale[j]
            bh_sep_in = bh_all_hsml[inds_in][-1] * scale[j]
            bh_mdot_in = bh_all_mdot[inds_in][-1]

            # recheck snapshot cut
            # inds_out = np.where((bh_all_partids == idn_o) & (bh_all_snapshots < snap))[0]
            inds_out = (bh_all_partids == idn_o) & (bh_all_snapshots < snap)
            if (np.count_nonzero(inds_out) < snapshot_cut) and (mso < 1e6):
                continue

            # get the separation for this black hole
            # bh_sep_out = bh_all_hsml[inds_out[-1]]*scale[j]
            bh_sep_out = bh_all_hsml[inds_out][-1] * scale[j]
            bh_mdot_out = bh_all_mdot[inds_out][-1]

            # find index for final black hole in all bhs catalog
            ind_bh_out = np.where((bh_all_partids == idn_o) & (bh_all_snapshots == snap))[0][0]

            # add coordinates
            coordinates_out.append(list(bh_all_coordinates[ind_bh_out]))

            # take the max of the intial separations
            separation_out.append(np.max([bh_sep_in, bh_sep_out]))

            # Take the sum of the accretion rates
            mdot_out.append(bh_mdot_in + bh_mdot_out)

            # add merger index
            merger_ind_out.append(uni_mergers[j])

            # add all ids involved
            ids_old_out.append([ido_i, ido_o, ido_o])
            ids_new_out.append([idn_i, idn_o, idn_o])

            # add masses involved
            masses_out.append([msi, mso, msi+mso])

            # add density profiles
            density_profiles_out.append([
                dens_profs['star_norm'][j],
                dens_profs['star_gamma'][j],
                dens_profs['gas_norm'][j],
                dens_profs['gas_gamma'][j],
                dens_profs['dm_norm'][j],
                dens_profs['dm_gamma'][j]
            ])

            redshift_out.append(redshift[j])

            # the following quantities will be added in groups of three
            # (one for each constituent and remnant black hole).
            # We create _trans lists to be a group of 3. We then add them to an overall list.
            subs_trans = []
            snaps_trans = []

            # velocity dispersion from calculation
            vel_disps_trans = []

            # velocity dispersion from the SubhaloVelDisp
            vel_disps_from_subs_trans = []

            # mass for remnant host galaxy
            stellar_mass_trans = []
            total_mass_trans = []

            # iterate through the velocity dispsersion halos (groups of 3)
            for k in np.arange(uni_vel_ind[j], uni_vel_ind[j]+3)[::-1]:
                sub = velocity_dispersions[k][3]
                snap = velocity_dispersions[k][2]

                subs_trans.append(sub)
                snaps_trans.append(snap)
                vel_disps_trans.append(velocity_dispersions[k][4])

                # get info from group catalog data

                # wind phase cells are counted in type 0,
                # even though they are really type 4 see docs
                # ind_gc = np.where((gc_snaps == snap) & (gc_subs == sub))[0][0]
                # stellar_mass = gc_SubhaloMassType[ind_gc][4] * 1e10/h
                # total_mass = np.sum(gc_SubhaloMassType[ind_gc] * 1e10/h)
                # vel disp from group catalog
                # vel_disps_from_subs_trans.append(gc_SubhaloVelDisp[ind_gc]*np.sqrt(3))
                ind_gc = ((gc_snaps == snap) & (gc_subs == sub))
                stellar_mass = gc_SubhaloMassType[ind_gc][0][4] * 1e10/h
                total_mass = np.sum(gc_SubhaloMassType[ind_gc][0] * 1e10/h)
                # vel disp from group catalog
                vel_disps_from_subs_trans.append(gc_SubhaloVelDisp[ind_gc][0] * np.sqrt(3))

                stellar_mass_trans.append(stellar_mass)
                total_mass_trans.append(total_mass)

            # add the three entry lists to the overall lists
            subs_out.append(subs_trans)
            snaps_out.append(snaps_trans)
            vel_disps_out.append(vel_disps_trans)
            vel_disps_from_subs_out.append(vel_disps_from_subs_trans)
            stellar_mass_out.append(stellar_mass_trans)
            total_mass_out.append(total_mass_trans)

        out_list = []
        # gather all data into single list (concatenate the quantity list)
        for i in range(len(merger_ind_out)):
            temp = [merger_ind_out[i]] + snaps_out[i] + subs_out[i]
            temp = temp + ids_old_out[i] + ids_new_out[i] + [mdot_out[i]] + masses_out[i]
            temp = temp + [redshift_out[i]] + [separation_out[i]] + coordinates_out[i]
            temp = temp + density_profiles_out[i] + vel_disps_out[i]
            temp = temp + stellar_mass_out[i] + total_mass_out[i]
            out_list.append(temp)
            #     [merger_ind_out[i]] + snaps_out[i] + subs_out[i] +
            #     ids_old_out[i] + ids_new_out[i] +
            #     mdot_out[i] + masses_out[i] +
            #     [redshift_out[i]] + [separation_out[i]] + coordinates_out[i] +
            #     density_profiles_out[i] + vel_disps_out[i] +
            #     stellar_mass_out[i] + total_mass_out[i]
            # )

        # write out data
        fname = self.core.fname_final_data()
        with open(fname, 'w') as f:
            f.write("merger\tsnap_prev_in\tsnapshot_prev_out\tsnapshot_fin_out\tsubhalo_prev_in\tsubhalo_prev_out\tsubhalo_fin_out\tid_old_prev_in\tid_old_prev_out\tid_old_fin_out\tid_new_prev_in\tid_new_prev_out\tid_new_fin_out\tmdot_sum\tmass_new_prev_in\tmass_new_prev_out\tmass_new_fin_out\tredshift\tseparation\tcoordinates_x\tcoordinates_y\tcoordinates_z\tstar_norm\tstar_gamma\tgas_norm\tgas_gamma\tdm_norm\tdm_gamma\tvel_disp_prev_in\tvel_disp_prev_out\tvel_disp_fin_out\tstellar_mass_prev_in\tstellar_mass_prev_out\tstellar_mass_fin_out\ttotal_mass_prev_in\ttotal_mass_prev_out\ttotal_mass_fin_out\n")

            # this ensures each data point is recorded with right dtype
            for item in out_list:
                for item2 in item:
                    f.write(str(item2) + '\t')
                f.write('\n')

        return
