from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
import scipy.signal as sci_sig

######### Opening images #########

hdulist = fits.open("TrabalhoComputacional/CH_PR100010_TG023201_TU2019-10-06T09-40-30_SCI_RAW_SubArray_V0000.fits")

# Header
header_subarray = hdulist[1].header

xlim = header_subarray['X_WINOFF']
ylim = header_subarray['Y_WINOFF']

# Images
images = hdulist[1].data
(nimage,ny,nx) = images.shape

######### Overscan #########

overscanleft = hdulist[7].data

# Applying Overscan to Images (Average over all points of the overscan)

images_b = np.zeros(images.shape) # Images without BIAS # IF USING ZEROS_LIKE, STRANGE BUG
for i in range(nimage):
	images_b[i] = images[i] - np.average(overscanleft[i])

######### Flat Field #########

hdulist_ff = fits.open("TrabalhoComputacional/CH_PR100010_TG023201_TU2019-10-06T09-40-00_SIM_TRU_FlatField_V0000.fits")

# Determining Flat Field subarray with the Object
ff = hdulist_ff[1].data
ff_subarray = ff[ylim:ylim+ny,xlim:xlim+nx]

# Applying Flat Field

images_f = np.zeros(images.shape) # Images without FF
for i in range(nimage):
	images_f[i] = images_b[i]/ff_subarray

######### Smooth Image #########

"""
N = 4
L = 1.
dh = 2.*L/(N-1)

X, Y = np.mgrid[-L:L+dh:dh,-L:L+dh:dh]

gauss = np.exp((X**2 + Y**2)/0.8**2)

for i in range(nimage):
	images_f[i] = sci_sig.convolve2d(images_f[i],gauss,mode='same')
"""

######### Center of Mass #########

R = np.zeros(nimage,dtype=object)

r_CM = 100 # Determined empirically

for i in range(nimage):
	square = images_f[i][100-r_CM:100+r_CM,100-r_CM:100+r_CM]
	M = np.sum(square)
	for j in range(2*r_CM):
		R[i] += np.array([np.sum(square[:,j]*j), np.sum(square[j,:]*j)])
	R[i] = (R[i]/M).astype(int)
	R[i] += (100-r_CM,100-r_CM)

######### Plotting #########

img_n = 0 # Image to represent

index = np.arange(0,nx,1)
x_i, y_i = np.meshgrid(index,index)

plt.figure(1)
for img_n in range(406):
	print(img_n)
	r_i = np.sqrt((x_i-R[img_n][0])**2 + (y_i-R[img_n][1])**2)

	ind =  (r_i >= 30) & (r_i <= 50)

	images_f[img_n] = images_f[img_n]-np.average(images_f[img_n][ind]) # Calculate and take out background noise

	images_f[img_n][(r_i >= 5) & (r_i <= 6)] += 100000 # Build a circle

	plt.imsave('imgs/cm_r_%s_img_%s.png' %(r_CM,img_n),arr=images_f[img_n],cmap='gray')
"""
	plt.title('Image %s with CM' %img_n)
	plt.imshow(images_f[img_n])
	plt.pause(0.1)
"""
############## Scharr Kernel Gradient ##############

from scipy import signal
import matplotlib.pyplot as plt
from scipy.ndimage.filters import gaussian_filter
from scipy.signal import medfilt


scharr = np.array([[ -3-3j, 0-10j,  +3 -3j],
                   [-10+0j, 0+ 0j, +10 +0j],
                   [ -3+3j, 0+10j,  +3 +3j]]) # Gx + j*Gy

fig, (ax_orig, ax_mag, ax_ang) = plt.subplots(3, 1, figsize=(6, 15))

ax_orig.set_title('Original')
ax_orig.set_axis_off()

ax_mag.set_title('Gradient magnitude')
ax_mag.set_axis_off()

ax_ang.set_title('Gradient orientation')
ax_ang.set_axis_off()

sigma = 1.5

def rgb2gray(rgb):
    return np.dot(rgb[...,:3], [0.299, 0.587, 0.114])

for i in range(406):
	print(i)

	images_f[i] = gaussian_filter(images_f[i],sigma)

	grad = signal.convolve2d(images_f[i], scharr, boundary='symm', mode='same')
	
	plt.imsave('imgs/gradient_angle_img_%s.png' %i,arr=np.angle(grad),cmap='gray_r')

"""	ax_orig.imshow(images_f[i], cmap='gray')
	
	ax_mag.imshow(np.absolute(grad), cmap='gray')
	
	ax_ang.imshow(np.angle(grad), cmap='hsv') # hsv is cyclic, like angles

	plt.pause(0.1)"""




#plt.show()
