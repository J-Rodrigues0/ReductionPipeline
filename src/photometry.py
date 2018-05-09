from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from time import clock

t0 = clock()

######### Opening images #########

hdulist = fits.open("../../TrabalhoComputacional/CH_PR100010_TG023201_TU2019-10-06T09-40-30_SCI_RAW_SubArray_V0000.fits")

# Header
header_subarray = hdulist[1].header
#print(header_subarray)

xlim = header_subarray['X_WINOFF']
ylim = header_subarray['Y_WINOFF']

# Images
images = hdulist[1].data
(nimage,ny,nx) = images.shape

######### Overscan #########

overscanleft = hdulist[7].data

# Applying Overscan to Images (Average over all points of the overscan)

images_b = np.zeros(images.shape) # Images without BIAS
for i in range(nimage):
	images_b[i] = images[i] - np.average(overscanleft[i])

######### Flat Field #########

hdulist_ff = fits.open("../../TrabalhoComputacional/CH_PR100010_TG023201_TU2019-10-06T09-40-00_SIM_TRU_FlatField_V0000.fits")

# Determining Flat Field subarray with the Object
ff = hdulist_ff[1].data
ff_subarray = ff[ylim:ylim+ny,xlim:xlim+nx]

# Applying Flat Field

images_f = np.zeros(images.shape) # Images without FF
for i in range(nimage):
	images_f[i] = images_b[i]/ff_subarray

######### Filtering #########

from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter

# Applying a median filter to the images to remove cosmic ray errors
# Applying a gaussian filter to remove gaussian noise (preparing for canny edge detection)


images_filt = np.zeros(images.shape) # Filtered images
for i in range(nimage):
	images_filt[i] = medfilt(images_f[i])

######### Center of Mass Calculation #########

R = np.zeros(nimage,dtype=object)

r_CM = 20

for i in range(nimage):
	square = images_f[i][100-r_CM:100+r_CM,100-r_CM:100+r_CM]
	M = np.sum(square)
	for j in range(2*r_CM):
		R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
	R[i] = (R[i]/M).astype(int)
	R[i] += (100-r_CM,100-r_CM)

######### Canny Edge Detection #########

from canny_edge_detection import canny_detect

sigma = 1.1
T = 3500
t = T*0.3

target_indx = canny_detect(images_filt[0],R[0],sigma,t,T) # Indexes of the target's area

######### Photometry #########

index = np.arange(0,nx,1)
x_i, y_i = np.meshgrid(index,index)

r_min = 25

background_mode = np.zeros(nimage)
background_avg = np.zeros(nimage)

N_counts = np.zeros(nimage)
for i in range(nimage):
	img = images_f[i]

	r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

	hist, bins = np.histogram(img[(r_i > r_min)],bins=1000)

	background_mode[i] = np.min(bins[np.where(hist == np.max(hist))])

	background_avg[i] = np.average(img[(r_i > r_min)])
	
	img -= background_mode[i]
	#img -= background_avg[i]

	N_counts[i] = np.sum(img[target_indx[0] - 100 + R[i][1],target_indx[1] - 100 + R[i][0]])

print(clock()-t0)

Flux = N_counts

img_pts = np.arange(0,nimage,1)

plt.figure(1)
plt.scatter(img_pts,Flux,color='k',marker='.')
plt.ylabel('Flux (Number of Counts)')
plt.xlabel('n')

plt.figure(2)
plt.scatter(img_pts,background_mode,color='k',marker='.')
plt.ylabel('Background Mode')
plt.xlabel('n')

plt.figure(3)
plt.scatter(img_pts,background_avg,color='k',marker='.')
plt.ylabel('Background Average')
plt.xlabel('n')

plt.show()

"""
# Calculate dispersion
dispersion = abs((Flux - np.roll(Flux,1))/(np.max(Flux)-np.min(Flux))*100)
plt.figure(4)
plt.scatter(img_pts,dispersion,color='k',marker='.')
plt.ylabel('Dispersion (%) (% of max(Flux) - min(Flux))')
plt.xlabel('n')
"""
# Calculate standard deviation

N = 50
std_dev = np.std(Flux[:N])/np.mean(Flux[:N])*100

print('Standard deviation of first %s points is: %s' %(N,round(std_dev,3)) + '%')

##### Optimization #####

T_vals = np.arange(1000,20000,400)
#T_vals = np.arange(2000,6000,50) # MINIMO DE DISPERSAO PARA T ~= 3500
dispersion = np.zeros(len(T_vals))
max_Flux = np.zeros(len(T_vals))
for k in range(len(T_vals)):
	T = T_vals[k]
	print(T)

	######### Center of Mass Calculation #########

	R = np.zeros(nimage,dtype=object)

	for i in range(nimage):
		square = images_f[i][80:120,80:120]
		M = np.sum(square)
		for j in range(40):
			R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
		R[i] = (R[i]/M).astype(int)
		R[i] += (80,80)

	######### Canny Edge Detection #########

	from canny_edge_detection import canny_detect

	sigma = 0.5
	t = T/3

	target_indx = canny_detect(images_filt[0],R[0],sigma,t,T) # Indexes of the target's area

	######### Photometry #########

	index = np.arange(0,nx,1)
	x_i, y_i = np.meshgrid(index,index)

	r_min = 25

	background_mode = np.zeros(nimage)
	background_avg = np.zeros(nimage)

	N_counts = np.zeros(nimage)
	for i in range(nimage):
		img = images_f[i].copy()

		r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

		background_avg[i] = np.average(img[(r_i > r_min)])

		hist, bins = np.histogram(img[(r_i > r_min)],bins=1000)

		background_mode[i] = np.min(bins[np.where(hist == np.max(hist))])
		
		img -= background_mode[i]
		img -= background_avg[i]

		N_counts[i] = np.sum(img[target_indx[0] - 100 + R[i][1],target_indx[1] - 100 + R[i][0]])

	Flux = N_counts/target_indx.size

	# Calculate dispersion
	dispersion[k] = np.std(Flux[:N])/np.mean(Flux[:N])*100
	max_Flux[k] = np.average(Flux)

plt.figure(5)
plt.plot(T_vals,dispersion,color='k')
plt.ylabel('Dispersion (%)')
plt.xlabel('Higher Threshold values')
plt.xticks(np.arange(1000,21000,2000))

plt.figure(1213)
plt.plot(T_vals,max_Flux,color='k')
plt.ylabel('Average of Flux')
plt.xlabel('Higher Threshold values')
plt.xticks(np.arange(1000,21000,2000))


S_vals = np.arange(0.1,2.,0.05) # MINIMO DE DISPERSÃO PARA 0.5 SIGMA\
dispersion = np.zeros(len(S_vals))
for k in range(len(S_vals)):
	sigma = S_vals[k]
	print(sigma)

	######### Center of Mass Calculation #########

	R = np.zeros(nimage,dtype=object)

	for i in range(nimage):
		square = images_f[i][80:120,80:120]
		M = np.sum(square)
		for j in range(40):
			R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
		R[i] = (R[i]/M).astype(int)
		R[i] += (80,80)

	######### Canny Edge Detection #########

	from canny_edge_detection import canny_detect

	T = 3500
	t = T/3

	target_indx = canny_detect(images_filt[0],R[0],sigma,t,T) # Indexes of the target's area

	######### Photometry #########

	index = np.arange(0,nx,1)
	x_i, y_i = np.meshgrid(index,index)

	r_min = 25

	background_mode = np.zeros(nimage)
	background_avg = np.zeros(nimage)

	N_counts = np.zeros(nimage)
	for i in range(nimage):
		img = images_f[i].copy()

		r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

		background_avg[i] = np.average(img[(r_i > r_min)])

		hist, bins = np.histogram(img[(r_i > r_min)],bins=1000)

		background_mode[i] = np.min(bins[np.where(hist == np.max(hist))])
		
		img -= background_mode[i]
		img -= background_avg[i]

		N_counts[i] = np.sum(img[target_indx[0] - 100 + R[i][1],target_indx[1] - 100 + R[i][0]])

	Flux = N_counts/target_indx.size

	# Calculate dispersion
	dispersion[k] = np.std(Flux[:N])/np.mean(Flux[:N])*100

plt.figure(6)
plt.plot(S_vals,dispersion,color='k')
plt.ylabel('Dispersion (%)')
plt.xlabel('Sigma values')
plt.xticks(np.arange(0.1,2.1,0.1))


T = 4000

t_vals = np.arange(0.1,1,0.5) # MINIMO DE DISPERSÃO PARA 0.5 SIGMA\
dispersion = np.zeros(len(t_vals))
for k in range(len(t_vals)):
	sigma = S_vals[k]
	print(sigma)

	######### Center of Mass Calculation #########

	R = np.zeros(nimage,dtype=object)

	for i in range(nimage):
		square = images_f[i][80:120,80:120]
		M = np.sum(square)
		for j in range(40):
			R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
		R[i] = (R[i]/M).astype(int)
		R[i] += (80,80)

	######### Canny Edge Detection #########

	from canny_edge_detection import canny_detect

	
	t = T*t_vals[k]

	target_indx = canny_detect(images_filt[0],R[0],sigma,t,T) # Indexes of the target's area

	######### Photometry #########

	index = np.arange(0,nx,1)
	x_i, y_i = np.meshgrid(index,index)

	r_min = 25

	background_mode = np.zeros(nimage)
	background_avg = np.zeros(nimage)

	N_counts = np.zeros(nimage)
	for i in range(nimage):
		img = images_f[i].copy()

		r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

		background_avg[i] = np.average(img[(r_i > r_min)])

		hist, bins = np.histogram(img[(r_i > r_min)],bins=1000)

		background_mode[i] = np.min(bins[np.where(hist == np.max(hist))])
		
		img -= background_mode[i]
		img -= background_avg[i]

		N_counts[i] = np.sum(img[target_indx[0] - 100 + R[i][1],target_indx[1] - 100 + R[i][0]])

	Flux = N_counts/target_indx.size

	# Calculate dispersion
	dispersion[k] = np.std(Flux[:N])/np.mean(Flux[:N])*100

plt.figure(7)
plt.plot(t_vals,dispersion,color='k')
plt.ylabel('Dispersion (%)')
plt.xlabel('Lower Threshold values (fraction of Higher Threshold)')
plt.xticks(np.arange(0.1,1.1,0.25))
"""
"""
r_CM_vals = np.arange(10,80,5)

dispersion = np.zeros(len(r_CM_vals))
for k in range(len(r_CM_vals)):
	sigma = 0.6

	######### Center of Mass Calculation #########
	r_CM = r_CM_vals[k]
	print(r_CM)

	R = np.zeros(nimage,dtype=object)

	for i in range(nimage):
		square = images_f[i][100-r_CM:100+r_CM,100-r_CM:100+r_CM]
		M = np.sum(square)
		for j in range(2*r_CM):
			R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
		R[i] = (R[i]/M).astype(int)
		R[i] += (100-r_CM,100-r_CM)

	######### Canny Edge Detection #########

	from canny_edge_detection import canny_detect

	T = 4000
	t = T*0.3

	target_indx = canny_detect(images_filt[0],R[0],sigma,t,T) # Indexes of the target's area

	######### Photometry #########

	index = np.arange(0,nx,1)
	x_i, y_i = np.meshgrid(index,index)

	r_min = 25

	background_mode = np.zeros(nimage)
	background_avg = np.zeros(nimage)

	N_counts = np.zeros(nimage)
	for i in range(nimage):
		img = images_f[i].copy()

		r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

		background_avg[i] = np.average(img[(r_i > r_min)])

		hist, bins = np.histogram(img[(r_i > r_min)],bins=1000)

		background_mode[i] = np.min(bins[np.where(hist == np.max(hist))])
		
		img -= background_mode[i]
		img -= background_avg[i]

		N_counts[i] = np.sum(img[target_indx[0] - 100 + R[i][1],target_indx[1] - 100 + R[i][0]])

	Flux = N_counts/target_indx.size

	# Calculate dispersion
	dispersion[k] = np.std(Flux[:N])/np.mean(Flux[:N])*100

plt.figure(8)
plt.plot(r_CM_vals,dispersion,color='k')
plt.ylabel('Dispersion (%)')
plt.xlabel('Square side used for CM calculation')
plt.xticks(np.arange(10,80,10))


r_min_vals = np.arange(10,80,5)
dispersion = np.zeros(len(r_min_vals))
for k in range(len(r_min_vals)):
	sigma = 0.6


	######### Center of Mass Calculation #########
	r_CM = 20

	R = np.zeros(nimage,dtype=object)

	for i in range(nimage):
		square = images_f[i][100-r_CM:100+r_CM,100-r_CM:100+r_CM]
		M = np.sum(square)
		for j in range(2*r_CM):
			R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
		R[i] = (R[i]/M).astype(int)
		R[i] += (100-r_CM,100-r_CM)

	######### Canny Edge Detection #########

	from canny_edge_detection import canny_detect

	T = 4000
	t = T*0.3

	target_indx = canny_detect(images_filt[0],R[0],sigma,t,T) # Indexes of the target's area

	######### Photometry #########

	index = np.arange(0,nx,1)
	x_i, y_i = np.meshgrid(index,index)

	r_min = r_min_vals[k]
	print(r_min)

	background_mode = np.zeros(nimage)
	background_avg = np.zeros(nimage)

	N_counts = np.zeros(nimage)
	for i in range(nimage):
		img = images_f[i].copy()

		r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

		background_avg[i] = np.average(img[(r_i > r_min)])

		hist, bins = np.histogram(img[(r_i > r_min)],bins=1000)

		background_mode[i] = np.min(bins[np.where(hist == np.max(hist))])
		
		img -= background_mode[i]
		img -= background_avg[i]

		N_counts[i] = np.sum(img[target_indx[0] - 100 + R[i][1],target_indx[1] - 100 + R[i][0]])

	Flux = N_counts/target_indx.size

	# Calculate dispersion
	dispersion[k] = np.std(Flux[:N])/np.mean(Flux[:N])*100

plt.figure(9)
plt.plot(r_CM_vals,dispersion,color='k')
plt.ylabel('Dispersion (%)')
plt.xlabel('Radius used for Background calculation')
plt.xticks(np.arange(10,80,10))

plt.show()