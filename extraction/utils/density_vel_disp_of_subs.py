"""
MAIN PURPOSE: calculated density profiles and velocity dispersion for merger host galaxies
"""

import numpy as np 
import h5py
import os
from scipy.optimize import curve_fit

h=0.704

def fit_func(x,a,b):
	"""
	Power law fit for the density profile
	"""
	return a*x**-b

class DensityProfVelDisp:
	"""
	DensityProfVelDisp calculates density profiles and velocity dispersions for the mergers. It gets density profiles for all particle types in remnant black hole host galaxies. Stellar velocity dispersions are calcualted for all merger-related galaxies. If a fit does not converge, the merger is no longer considered part of our catalog. 

		Note: when downloading all the subhalos, some downloads may not have been perfect meaning the files will not open. This should be a very small number. If this is the case, the code will error and stop running. you can see the last subhalo that the code tried to open. Use `download_single.py` for these files. 

		attributes:
			:param directory - (str) - directory to work in

		methods:
			find_fit
			gather_info_from_mergers
			gather_info_from_subs_with_bhs
			fit_main

	"""

	def __init__(self, directory):
		self.directory = directory

		if 'density_profiles.txt' in os.listdir(self.directory) and 'velocity_dispersions.txt' in os.listdir(self.directory):
			self.needed = False

		else:
			self.needed = True


	def find_fit(self, coordinates, masses, sub_cm, scale):
		"""
		Function used to determine fit. Used for all particle types.
		"""

		#radius from CoM
		radius = np.sqrt(np.sum((coordinates - sub_cm)**2, axis=1))*scale*1e3 #comoving to physical -- kpc to pc

		#put in structured array for sorting
		all_particles = np.core.records.fromarrays([radius, masses], dtype=[('rad', np.dtype(float)),('mass', np.dtype(float))])

		all_particles = np.sort(all_particles, order=('rad',))

		radial_bins_edges = np.logspace(np.log10(all_particles['rad'][0]-1), np.log10(all_particles['rad'][-1]+1), 100)

		#figure out which bin each particle is in
		bin_number_for_each_paricle = np.searchsorted(radial_bins_edges, all_particles['rad'])
		unique_bins, bin_count = np.unique(bin_number_for_each_paricle, return_counts=True)

		#make sure there are at least 8 bins with 4 or more particles in them.
		try:
			inds_bins = np.where(bin_count>=4)[0][0:8]
		except IndexError:
			run_vel_disp = False
			return [], run_vel_disp

		unique_bins = unique_bins[inds_bins]

		#get necessary bin edges for volume only for eight bins tested
		radial_bins_inner_edges = radial_bins_edges[inds_bins]
		radial_bins_outer_edges = radial_bins_edges[inds_bins+1]

		#use bin centers for radial profile calculation
		radial_bin_centers = (radial_bins_outer_edges + radial_bins_inner_edges)/2.

		volume_bin = 4/3.*np.pi*(radial_bins_outer_edges**3 - radial_bins_inner_edges**3)
		#density_bin = np.zeros(8)

		# find bin densities
		density_bin = np.zeros(len(radial_bin_centers))
		for i, bin in enumerate(unique_bins):
			#which particles are in this bin
			inds_bin = np.where(bin_number_for_each_paricle == bin)[0]
			density_bin[i] = np.sum(all_particles['mass'][inds_bin])/volume_bin[i]

		#try fit
		var, cov = curve_fit(fit_func, radial_bin_centers[0:8], density_bin[0:8])
		return var, True

	def gather_info_from_mergers(self):
		"""
		Gets merger information.
		"""
		with h5py.File('bhs_mergers_new.hdf5', 'r') as mergers:
			#scale factor is needed to scale coordinates
			merger_time = mergers['time'][:]

		return merger_time

	def gather_info_from_subs_with_bhs(self):
		"""
		Gets information from galaxies with bhs.
		"""

		with h5py.File('subs_with_bhs.hdf5', 'r') as gc:
			subs_gc = gc['SubhaloID'][:]
			snaps_gc = gc['Snapshot'][:]
			subhalo_cm_gc = gc['SubhaloCM'][:]

		return subs_gc, snaps_gc, subhalo_cm_gc

	def fit_main(self):
		"""
		Main function controlling fitting density profiles and getting velocity dispersions.
		"""

		#get information about subs downloaded
		subs_downloaded = np.genfromtxt('snaps_and_subs_needed.txt', dtype=int)

		#get other information needed
		merger_time = self.gather_info_from_mergers()
		subs_gc, snaps_gc, subhalo_cm_gc = self.gather_info_from_subs_with_bhs()
		

		vel_disp_out = []
		density_profile_out = []

		#this figures out the remnant host halo. Order in ``snaps_and_subs_needed.txt`` is 3,2,1
		uni_mergers, uni_index = np.unique(subs_downloaded[:,0], return_index=True)

		#bin selection arbitrary but did not affect results much
		bins = 100
		print(len(uni_index), 'to go through')

		#idea is to only get velocity dispersion of other subhalos if main halo can get the fit. 
		#if fit does not work , the merger is not considered
		for num, m_start_ind in enumerate(uni_index):
			run_vel_disp = True

			#this scrolls through the three subhalos per merger
			for index in np.arange(m_start_ind, m_start_ind+3)[::-1]:
				
				if run_vel_disp ==False:
					continue

				#extract galaxy info
				row = subs_downloaded[index]
				snap = row[2]
				sub = row[3]

				#merger index
				m = row[0]

				#final, prev_in, or prev_out
				which = row[1]
				print(snap, sub, num)
				if which == 3: 
					#scale factor
					scale = merger_time[m]

					ind = np.where((snaps_gc == snap) & (subs_gc == sub))[0][0]

					#center of mass of galaxy
					sub_cm = subhalo_cm_gc[ind]
			
					with h5py.File('%i/%i_sub_cutouts/cutout_%i_%i.hdf5'%(snap, snap, snap, sub), 'r') as f:
					#for local testing
					#with h5py.File('cutout_%i_%i.hdf5'%(snap, sub), 'r') as f:

						######## stars firts ########
						stars = f['PartType4']

						#if GFM_StellarFormationTime < 0.0, it is a wind particle, not a star.
						keep = np.where(stars['GFM_StellarFormationTime'][:] >= 0.0)[0]

						#confirm there are 80 star particles (there should be from previous analysis)
						if len(keep) < 80 and which == 3:
							run_vel_disp = False
							continue

						#try the profile. If a fit cannot be done, then catch error and go to next merger
						try:
							var, run_vel_disp = self.find_fit(stars['Coordinates'][:][keep], stars['Masses'][:][keep]*1e10/h, sub_cm, scale)
							if run_vel_disp == False:
								continue

						except RuntimeError:
							run_vel_disp = False
							continue

						#store gamma value for stars
						star_gamma = var[1]

						######## gas second ########

						#sometimes halos are completely missing gas. We do not consider these as part of the catalog.
						try:
							gas = f['PartType0']
						except KeyError:
							run_vel_disp = False
							continue

						#make sure 80 gas cells
						if len(gas['Coordinates'][:]) < 80:
							run_vel_disp = False
							continue

						try:
							var, run_vel_disp = self.find_fit(gas['Coordinates'][:], gas['Masses'][:]*1e10/h, sub_cm, scale)
							if run_vel_disp == False:
								continue

						except RuntimeError:
							run_vel_disp = False
							continue
						gas_gamma = var[1]

						######## dm third ########
						dm = f['PartType4']

						#make sure 300 dm particles
						if len(dm['Coordinates'][:]) < 300:
							run_vel_disp = False
							continue

						try:
							#dm particles all have the same mass
							var, run_vel_disp = self.find_fit(dm['Coordinates'][:], np.full((len(dm['Coordinates'][:]),),6.3e6), sub_cm, scale)
							if run_vel_disp == False:
								continue

						except RuntimeError:
							run_vel_disp = False
							continue
						dm_gamma = var[1]

						#add data to list if all try/excepts are succesful
						density_profile_out.append([m, which, snap, sub, star_gamma, gas_gamma, dm_gamma])

						#get velocity dispersion
						if run_vel_disp:
							vel_disp_out.append([m, which, snap, sub, np.std(np.sqrt(np.sum(stars['Velocities'][:]**2, axis=1)))])

				else:
					#get only velocity dispersion for constituent halos
					if run_vel_disp:
						with h5py.File('%i/%i_sub_cutouts/cutout_%i_%i.hdf5'%(snap, snap, snap, sub), 'r') as f:
						#for testing in local directories
						#with h5py.File('cutout_%i_%i.hdf5'%(snap, sub), 'r') as f:
							vel_disp_out.append([m, which, snap, sub, np.std(np.sqrt(np.sum(f['PartType4']['Velocities'][:]**2, axis=1)))])

		### prep all the lists for read out and read out to files ###

		density_profile_out = np.asarray(density_profile_out).T
		vel_disp_out = np.asarray(vel_disp_out).T

		density_profile_out = [density_profile_out[i] for i in range(len(density_profile_out))]
		vel_disp_out = [vel_disp_out[i] for i in range(len(vel_disp_out))]
				
		density_profile_out = np.core.records.fromarrays(density_profile_out, dtype=[('merger', np.dtype(int)),('which', np.dtype(int)),('snap', np.dtype(int)),('sub', np.dtype(int)),('star_gamma', np.dtype(float)), ('gas_gamma', np.dtype(float)), ('dm_gamma', np.dtype(float))])

		np.savetxt('density_profiles.txt', density_profile_out, fmt='%i\t%i\t%i\t%i\t%.18e\t%.18e\t%.18e', header = 'm\twhich\tsnap\tsub\tstar_gamma\tgas_gamma\tdm_gamma')

		vel_disp_out = np.core.records.fromarrays(vel_disp_out, dtype=[('merger', np.dtype(int)),('which', np.dtype(int)),('snap', np.dtype(int)),('sub', np.dtype(int)),('vel_disp', np.dtype(float))])

		np.savetxt('velocity_dispersions.txt', vel_disp_out, fmt='%i\t%i\t%i\t%i\t%.18e', header = 'm\twhich\tsnap\tsub\tvel_disp')

		return






