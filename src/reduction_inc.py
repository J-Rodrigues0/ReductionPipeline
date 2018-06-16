from astropy.io import fits
import numpy as np
import matplotlib.pyplot as plt

def overscan(hdulist,images):
	"""
	Applies the overscan to the images by performing an average of 
	all the points of the overscan and subtracting it from each image.
	"""

	# Open overscan data
	overscanleft = hdulist[7].data
	(nimage,nx,ny) = images.shape

	# Images without BIAS
	images_b = np.zeros(images.shape)
	for i in range(nimage):
		images_b[i] = images[i] - np.average(overscanleft[i])

	return images_b

def flatfield(hdulist,images_b,flat_path):
	"""
	Applies the flatfield to the images by rerieving the flatfield
	array and dividing the images by the data.
	"""

	# Open flatfield data
	hdulist_ff = fits.open(flat_path)
	(nimage,nx,ny) = images_b.shape

	# Fits header to retrieve xlim and ylim
	header_subarray = hdulist[1].header
	xlim = header_subarray['X_WINOFF']
	ylim = header_subarray['Y_WINOFF']

	# Determining Flat Field subarray with the Object
	ff = hdulist_ff[1].data
	ff_subarray = ff[ylim:ylim+ny,xlim:xlim+nx]

	# Images without FF
	images_f = np.zeros(images_b.shape) 
	for i in range(nimage):
		images_f[i] = images_b[i]/ff_subarray

	return images_f

def calc_CM(images_f):
	"""
	Computes the center of mass (CM) of the target.
	Selects a square area around the center of the images and computes
	the CM in that area.
	"""

	# Initialize CM positions array
	(nimage,nx,ny) = images_f.shape
	R = np.zeros(nimage,dtype=object)

	# Side of the square
	r_CM = 20 # Determined empirically

	# Iterate through the images and compute each CM
	for i in range(nimage):
		# Select square area
		square = images_f[i][100-r_CM:100+r_CM,100-r_CM:100+r_CM]
		M = np.sum(square)

		for j in range(2*r_CM):
			# Accumulate data from each point of the square
			R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
		# Divide by the sum of the data inside the square
		R[i] = (R[i]/M).astype(int)

		# Shift to image center (as opposed to square center)
		R[i] += (100-r_CM,100-r_CM)

	return R

def calc_TARGET(images_f,R):
	"""
	Computes the area of the target star inside the image.
	Applies the canny edge detection algorithm in canny_edge_detection.py
	"""

	# Import canny edge algorithm
	from canny_edge_detection import canny_detect

	sigma = 1.85 # Determined empirically
	T = 3500 # Determined empirically
	t = T*0.5 # Determined empirically

	# Apply algorithm
	target = canny_detect(images_f[0],R[0],sigma,t,T)

	return target

def calc_background(images_f,R):
	"""
	Computes the background mode and mean to later be subtracted from the
	target's values in order to perform the photometry.
	The background is considered to be any point outside a circle of radius
	r_bkg centered on the CM of the target.
	"""

	# Initialize x_i and y_i position arrays
	(nimage,nx,ny) = images_f.shape
	index = np.arange(0,nx,1)
	x_i, y_i = np.meshgrid(index,index)

	r_bkg = 25 # Determined empirically

	# Initialize mode and mean arrays
	bkg_mode = np.zeros(nimage)
	bkg_mean = np.zeros(nimage)

	for i in range(nimage):
		# Initialize distance from CM array
		r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

		# Perform histogram to compute mode
		hist, bins = np.histogram(images_f[i][(r_i > r_bkg)],bins=1000)

		# Mode is the minimum between the values of greatest appearance
		bkg_mode[i] = np.min(bins[np.where(hist == np.max(hist))])

		# Mean is the average of the points
		bkg_mean[i] = np.average(images_f[i][(r_i > r_bkg)])

	return bkg_mode, bkg_mean

def photometry(images_f,R,target,bkg_mode,**kwargs):
	(nimage,nx,ny) = images_f.shape

	""" 
	Select background mode and mean of the background to subtract from the images.
	The mean subtraction is optional since it only tries to remove the noise from the Pacific Anomaly
	but also increases the overall dispersion of the data.
	"""

	# Initialize flux array
	flux = np.zeros(nimage)

	try:
		bkg_mean = kwargs['bkg_mean']
	except:
		print("NOT SUBTRACTING BACKGROUND MEAN")

	# Iterate through the images and compute flux
	for i in range(nimage):
		# Subtract mode
		images_f[i] -= bkg_mode[i]

		# Subtract mean if selected
		try:
			images_f[i] -= bkg_mean[i]
		except:
			pass

		# Sum points belonging to the target
		flux[i] = np.sum(images_f[i][target[0] - 100 + R[i][1],target[1] - 100 + R[i][0]])

	return flux

def calc_dispersion(flux,Ni,Nf):
	"""
	Compute dispersion of the flux data.
	"""

	# Calculate standard deviation and mean
	std_dev = np.std(flux[Ni:Nf])
	mean = np.mean(flux[Ni:Nf])

	# Compute dispersion
	dispersion = std_dev/mean*100 # In %

	# dispersion_6h = (dispersion(%)/100)/sqrt(6*60)*1e6 (ppm)
	dispersion_6h = dispersion/(100*np.sqrt(6*60))*1e6;

	return mean,dispersion,dispersion_6h

def treat_data(flux,out_path,**kwargs):
	"""
	Treats the data to plot the flux and optionally, the background mean and mode.
	Saves the data to a data.txt file in the out_path folder.
	"""

	fits_name = kwargs['fits_name']
	flat_name = kwargs['flat_name']

	# Build header of the file
	header =  'PIPELINE DATA\n'
	header += 'FITS FILE: %s\n' %fits_name
	header += 'FLATFIELD FILE: %s\n' %flat_name

	img_pts = np.arange(0,len(flux),1)

	# Plot flux figure
	plt.figure(1)
	plt.scatter(img_pts,flux,color='k',marker='.',label='Flux data')

	# Compute relevant dispersion data
	Ni_vals = [0,260]
	Nf_vals = [60,320]
	lines = ['--','-.']
	for Ni,Nf,line in zip(Ni_vals,Nf_vals,lines):
		mean,dispersion, dispersion_6h = calc_dispersion(flux,Ni,Nf)

		plt.plot(img_pts[Ni:Nf],np.ones(Nf-Ni)*(mean + dispersion*mean/100),color='r',
					linestyle=line,label='Dispersion = %.3f' %dispersion + '%')
		plt.plot(img_pts[Ni:Nf],np.ones(Nf-Ni)*(mean - dispersion*mean/100),color='r',linestyle=line)

		header += 'DISPERSION FOR 6 HOURS AVERAGE FOR POINTS [%s,%s] (ppm): %.3f\n' %(Ni,Nf,dispersion_6h)

	data = np.empty([len(flux),3])

	plt.grid(color='gray', linestyle='--', linewidth=0.5)
	plt.ylabel('Flux (Number of Counts)')
	plt.xlabel('n')
	plt.legend()
	plt.savefig(out_path + '/flux.png',dpi=400,bbox_inches='tight')
	
	# Optional background mean and mode plot.

	try:
		bkg_mode = kwargs['bkg_mode']

		plt.figure(2)
		plt.scatter(img_pts,bkg_mode,color='k',marker='.')
		plt.grid(color='gray', linestyle='--', linewidth=0.5)
		plt.ylabel('Background Mode')
		plt.xlabel('n')
		plt.savefig(out_path + '/bkg_mode.png',dpi=400,bbox_inches='tight')
	except:
		print("NOT PLOTTING BACKGROUND MODE DATA")

	try:
		bkg_mean = kwargs['bkg_mean']

		plt.figure(3)
		plt.scatter(img_pts,bkg_mean,color='k',marker='.')
		plt.grid(color='gray', linestyle='--', linewidth=0.5)
		plt.ylabel('Background Average')
		plt.xlabel('n')
		plt.savefig(out_path + '/bkg_mean.png',dpi=400,bbox_inches='tight')
	except:
		print("NOT PLOTTING BACKGROUND MEAN DATA")

	# Finish header
	header += 'FLUX\tBKG_MODE\tBKG_MEAN'

	# Save data
	np.savetxt(out_path + '/data.txt', kwargs['data'], fmt='%.18e', delimiter='\t', header=header, encoding=None)

	print("DATA EXPORTED TO: %s" %out_path)