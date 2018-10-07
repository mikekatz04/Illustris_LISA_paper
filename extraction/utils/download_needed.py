"""
MAIN PURPOSE: download all of the subhalos needed. 
"""

import h5py
import numpy as np
import os

from utils.generalfuncs import get, download_sub


class DownloadNeeded:
	"""
	DownloadNeeded downloads all of the subhalos necessary for host galaxy characterization of the mergers. For host galaxies post-merger, particle information is downloaded for gas, star, bh, and dm particles. For host galaxies pre-merger, particle information is downloaded for star and bh particles. 

		THIS CODE DOES NOT CHECK IF IT IS NEEDED. THE USER MUST KNOW STATUS USING ``completed_snaps_and_subs.txt`` COMPARED TO ``snaps_and_subs_needed.txt``.

		To use this file, you need to have a specific file structure within self.directory. 
			'%i/%i_sub_cutouts/'%(snap, snap). The files are then stored in the struture as '%i/%i_sub_cutouts/cutout_%i_%i.hdf5'%(snap, snap, snap, sub). 

			attributes:
				:param	ill_run - (int) - illustris run to use
				:param	directory - (str) - directory to work out of

	"""

	def __init__(self, ill_run=1, directory='./extraction'):
		self.ill_run = ill_run
		self.directory = directory

		#NO CHECK IF IT IS NEEDED

	def download_needed_subhalos(self):
		"""
		Download all the subhalos files needed. 
		"""
		base_url = "http://www.illustris-project.org/api/Illustris-%i/" %self.ill_run

		snap_subs_needed = np.genfromtxt('snaps_and_subs_needed.txt', skip_header=1, names=True, dtype=None)

		#need to keep track because this process will time out. 
		try:
			f_complete = open('completed_snaps_and_subs.txt', 'r+')
		except FileNotFoundError:
			f_complete = open('completed_snaps_and_subs.txt', 'w')

		#figure out where you left off
		try:
			start = len(f_complete.readlines())
		except io.UnsupportedOperation:
			start = 0

		print(start)
		print(len(snap_subs_needed[start:]))
		for i, row in enumerate(snap_subs_needed[start:]):
			which = row[1]
			snap = row[2]
			sub = row[3]

			#if which != 3, we only want stars and bhs
			#check if this file is already there
			if which != 3:
				if 'cutout_%i_%i.hdf5'%(snap, sub) in os.listdir('%i/%i_sub_cutouts/'%(snap, snap)):
					with h5py.File('%i/%i_sub_cutouts/cutout_%i_%i.hdf5'%(snap, snap, snap, sub), 'r') as f:
						if 'PartType4' in f:
							f_complete.write('%i\t%i\t%i\t%i\n'%(row[0], row[1], snap, sub))
							print('already there')
							continue
			
			#if which != 3, we only want stars and bhs
			if which !=3:
				cutout_request={'bhs':'all', 'stars':'Coordinates,Masses,Velocities,GFM_StellarFormationTime'}

			#if which == 3, get all particle data with quantities of interest
			else:
				cutout_request={'bhs':'all', 'gas':'Coordinates,Masses,Velocities,StarFormationRate', 'stars':'Coordinates,Masses,Velocities,GFM_StellarFormationTime', 'dm':'Coordinates'}

			cutout = download_sub(base_url, snap,sub, cutout_request=cutout_request)
			
			if '%i'%snap not in os.listdir(self.directory):
				os.mkdir('%i/'%snap)

			if '%i_sub_cutouts'%snap not in os.listdir(self.directory + '%i/'%snap):
				os.mkdir(self.directory + '%i/'%snap + '%i_sub_cutouts/'%snap)

			#move file for good organization
			os.rename(cutout, '%i/%i_sub_cutouts/cutout_%i_%i.hdf5'%(snap, snap, snap, sub))

			#for testing in local folder
			#os.rename(cutout, 'cutout_%i_%i.hdf5'%(snap, sub))

			#record what has been completed
			f_complete.write('%i\t%i\t%i\t%i\n'%(row[0], row[1], snap, sub))

			if (i+1) % 1 == 0:
				print(i + start + 1)

		return






