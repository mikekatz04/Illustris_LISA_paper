"""
Support functions for all the scripts in utils.
"""

import requests


#refer to illustris API
def get(path, params=None):
	"""
	This function retrieves files from the Illustris server. See http://www.illustris-project.org/data/docs/api/ for more information on this function.
	"""
	# make HTTP GET request to path
	headers = {"api-key":"3d6cce0a4f31c43ee502cfdeb2302ded"}
	r = requests.get(path, params=params, headers=headers)

	# raise exception if response code is not HTTP SUCCESS (200)
	r.raise_for_status()

	if r.headers['content-type'] == 'application/json':
		return r.json() # parse json responses automatically

	if 'content-disposition' in r.headers:
		filename = r.headers['content-disposition'].split("filename=")[1]
		with open(filename, 'wb') as f:
			f.write(r.content)
		return filename # return the filename string
	return r

def download_sub(base_url, snap,sub,cutout_request={}):
	find_url = base_url + "snapshots/" + str(int(snap)) + "/" + 'subhalos' + "/" + str(int(sub))
	find_url += "/" + "cutout.hdf5"

	return get(find_url,cutout_request)


