from astropy.io import fits
import numpy as np

def overscan(hdulist,images):
	overscanleft = hdulist[7].data
	(nimage,nx,ny) = images.shape

	# Images without BIAS

	images_b = np.zeros(images.shape) # Images without BIAS
	for i in range(nimage):
		images_b[i] = images[i] - np.average(overscanleft[i])

	return images_b

def flatfield(hdulist,images_b,flat_path):

	hdulist_ff = fits.open(flat_path)
	(nimage,nx,ny) = images_b.shape

	# Fits header to retrieve xlim and ylim
	header_subarray = hdulist[1].header
	xlim = header_subarray['X_WINOFF']
	ylim = header_subarray['Y_WINOFF']

	# Determining Flat Field subarray with the Object
	ff = hdulist_ff[1].data
	ff_subarray = ff[ylim:ylim+ny,xlim:xlim+nx]

	images_f = np.zeros(images_b.shape) # Images without FF
	for i in range(nimage):
		images_f[i] = images_b[i]/ff_subarray

	return images_f

def calc_CM(images_f):
	(nimage,nx,ny) = images_f.shape
	R = np.zeros(nimage,dtype=object)

	r_CM = 20 # Determined empirically

	for i in range(nimage):
		square = images_f[i][100-r_CM:100+r_CM,100-r_CM:100+r_CM]
		M = np.sum(square)
		for j in range(2*r_CM):
			R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
		R[i] = (R[i]/M).astype(int)
		R[i] += (100-r_CM,100-r_CM)

	return R

def calc_TARGET(images_f,R):
	from canny_edge_detection import canny_detect

	sigma = 1.1 # Determined empirically
	T = 4000 # Determined empirically
	t = T*0.3 # Determined empirically

	target = canny_detect(images_f[0],R[0],sigma,t,T)

	return target

def calc_background(images_f,R):
	(nimage,nx,ny) = images_f.shape
	index = np.arange(0,nx,1)
	x_i, y_i = np.meshgrid(index,index)

	r_bkg = 25 # Determined empirically

	background_mode = np.zeros(nimage)
	background_avg = np.zeros(nimage)

	for i in range(nimage):
		r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

		hist, bins = np.histogram(images_f[i][(r_i > r_bkg)],bins=1000)

		background_mode[i] = np.min(bins[np.where(hist == np.max(hist))])
		background_avg[i] = np.average(images_f[i][(r_i > r_bkg)])

	return background_mode, background_avg

def photometry(images_f,R,target,**kwargs):
	(nimage,nx,ny) = images_f.shape

	try:
		bkg_mode = kwargs['bkg_mode']
	except:
		print("NOT SUBTRACTING BACKGROUND MODE")

	try:
		bkg_mean = kwargs['bkg_mean']
	except:
		print("NOT SUBTRACTING BACKGROUND MEAN")

	flux = np.zeros(nimage)

	for i in range(nimage):

		try:
			images_f[i] -= bkg_mode[i]
		except:
			pass

		try:
			images_f[i] -= bkg_mean[i]
		except:
			pass

		flux[i] = np.sum(images_f[i][target[0] - 100 + R[i][1],target[1] - 100 + R[i][0]])

	return flux