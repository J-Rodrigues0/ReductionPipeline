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

######### Center of Mass Calculation #########

R = np.zeros(nimage,dtype=object)

r_CM = 20 # Determined empirically

for i in range(nimage):
	square = images_f[i][100-r_CM:100+r_CM,100-r_CM:100+r_CM]
	M = np.sum(square)
	for j in range(2*r_CM):
		R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
	R[i] = (R[i]/M).astype(int)
	R[i] += (100-r_CM,100-r_CM)

######### Filtering #########

from scipy.signal import medfilt
from scipy.ndimage.filters import gaussian_filter

# Applying a median filter to the images to remove cosmic ray errors
# Applying a gaussian filter to remove gaussian noise (preparing for canny edge detection)


images_filt = np.zeros(images.shape) # Filtered images
for i in range(nimage):
	images_filt[i] = medfilt(images_f[i])

######### Canny Edge Detection #########

from canny_edge_detection import canny_detect

sigma = 1.1 # Determined empirically
T = 4000 # Determined empirically
t = T*0.3 # Determined empirically

target_indx = canny_detect(images_f[0],R[0],sigma,t,T)

######### Photometry #########

index = np.arange(0,nx,1)
x_i, y_i = np.meshgrid(index,index)

r_bkg = 25 # Determined empirically

background_mode = np.zeros(nimage)
background_avg = np.zeros(nimage)

N_counts = np.zeros(nimage)
for i in range(nimage):
	img = images_f[i]

	r_i = np.sqrt((x_i-R[i][0])**2 + (y_i-R[i][1])**2)

	hist, bins = np.histogram(img[(r_i > r_bkg)],bins=1000)

	background_mode[i] = np.min(bins[np.where(hist == np.max(hist))])
	background_avg[i] = np.average(img[(r_i > r_bkg)])
	
	#img -= background_mode[i]
	img -= background_avg[i]

	N_counts[i] = np.sum(img[target_indx[0] - 100 + R[i][1],target_indx[1] - 100 + R[i][0]])

Flux = N_counts

print('Total time elapsed: %s' %(clock() - t0))

img_pts = np.arange(0,nimage,1)

plt.figure(1)
plt.scatter(img_pts,Flux,color='k',marker='.')
plt.ylabel('Flux (Number of Counts)')
plt.xlabel('n')
plt.show()

######### Plotting #########
"""
fig, axes = plt.subplots(2,2)

(ax1,ax2,ax3,ax4) = (axes[0][0], axes[0][1], axes[1][0], axes[1][1])

for img_n in range(406):
	ax1.set_title('Image %s with BIAS and FLAT FIELD' %img_n)
	ax1.imshow(images[img_n])

	ax2.set_title('Image %s without BIAS' %img_n)
	ax2.imshow(images_b[img_n])

	ax3.set_title('Image %s without FF' %img_n)
	ax3.imshow(images_f[img_n])

	ax4.set_title('Image %s Filtered' %img_n)
	ax4.imshow(images_filt[img_n])

	plt.pause(0.1)

plt.show()
"""