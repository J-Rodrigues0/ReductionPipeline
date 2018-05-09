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

sigma = 0.6 # Determined empirically
T = 10000 # Determined empirically
t = T*0.3 # Determined empirically

img = np.zeros(images.shape) # Filtered image
for i in range(nimage):
	img[i] = images_f[i].copy()

	target_indx = canny_detect(img[i],R[i],sigma,t,T)

	img[i][target_indx[0],target_indx[1]] = 100000

plt.figure(1)
plt.imshow(images_f[0],cmap='gray')
#plt.imsave('../report/figs/img_0.png',arr = images_f[0],cmap = 'gray')

plt.figure(2)
plt.imshow(img[0],cmap='gray')
#plt.imsave('../report/figs/img_0_canny.png',arr = img[0],cmap = 'gray')
plt.show()