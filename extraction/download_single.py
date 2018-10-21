"""
MAIN PURPOSE: this file provides an easy way to download a specific subhalo from the command line. After `running download_needed.py`, there may be subhalos files that have issues opening. This script allows the user to quickly redownload those specific files. 

	Positional arguments on the command line are the snapshot following by the subhalo. For example:
		python download_single.py (snap) (sub)

	Optional arguments:
		--ill_run - (int) - illustris run to download from. Default is 1. For example:
			python download_single.py (snap) (sub) --ill_run 1

		--move - just needs to be place on the command line. If so, it will return True. This moves the downloaded file within the structure used in `download_needed.py`. (%i/%i_sub_cutouts/cutout_%i_%i.hdf5) For example:

			python download_single.py (snap) (sub) --move

		The rest of the arguments just need to placed on the command line without any other option. You must put at least one of these. You can mix and match as desired. These options set the cutout request to the illustris server. See http://www.illustris-project.org/data/docs/api/ for more info. 

		--all - get all the particle information used in the paper
			cut_req = {'bhs':'all', 'gas':'Coordinates,Masses,Velocities,StarFormationRate', 'stars':'Coordinates,Masses,Velocities,GFM_StellarFormationTime', 'dm':'Coordinates'}

		--stars - get the star particle information
			cut_req['stars'] = 'Coordinates,Masses,Velocities,GFM_StellarFormationTime'

		--gas - get the gas cell information
			cut_req['gas'] = 'Coordinates,Masses,Velocities,StarFormationRate'

		--dm - get the dm particle information
			cut_req['dm'] = 'Coordinates'

		Example:
			python download_single.py (snap) (sub) --stars --gas 


"""
if __name__ == "__main__":

	import argparse
	import os
	from utils.generalfuncs import get, download_sub

	parser = argparse.ArgumentParser()
	parser.add_argument("snap", type=int, help="snapshot number to download from")
	parser.add_argument("sub", type=int, help="subhalo number to download from")
	parser.add_argument("--ill_run", type=int, default=1)
	parser.add_argument("--all", action="store_true")
	parser.add_argument("--stars", action="store_true")
	parser.add_argument("--gas", action="store_true")
	parser.add_argument("--dm", action="store_true")
	parser.add_argument("--move", action="store_true")
	args = parser.parse_args()

	if args.all==False and args.stars==False and args.gas == False and args.dm ==False:
		raise Exception('Need to specify at least one of --all, --start, --gas, or --dm.')

	# confirm download to user	
	print("You want to download subhalo %i from snapshot %i in Illustris run %i." %(args.sub, args.snap, args.ill_run))

	# set the cutout request based on command line input
	if args.all:
		cut_req = {'bhs':'all', 'gas':'Coordinates,Masses,Velocities,StarFormationRate', 'stars':'Coordinates,Masses,Velocities,GFM_StellarFormationTime', 'dm':'Coordinates'}

	else:
		cut_req={}
		if args.stars:
			cut_req['stars'] = 'Coordinates,Masses,Velocities,GFM_StellarFormationTime'
		if args.gas:
			cut_req['gas'] = 'Coordinates,Masses,Velocities,StarFormationRate'
		if args.dm:
			cut_req['dm'] = 'Coordinates'

	# download
	base_url = "http://www.illustris-project.org/api/Illustris-%i/" %args.ill_run
	cutout = download_sub(base_url,args.snap,args.sub, cut_req)

	# move file in 
	if args.move:
		os.rename(cutout, '%i/%i_sub_cutouts/cutout_%i_%i.hdf5'%(args.snap, args.snap, args.snap, args.sub))
	else:
		os.rename(cutout, 'cutout_%i_%i.hdf5'%(args.snap, args.sub))

	print('Downloaded subhalo %i from snapshot %i.'%(args.snap, args.sub))

