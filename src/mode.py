from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from time import clock
from scipy.optimize import curve_fit

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

images_b = np.zeros(images.shape) # Images without BIAS # IF USING ZEROS_LIKE, STRANGE BUG
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

######### Mode Calculation #########

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


def sin(x,*p):
	A, f, phs, DC = p
	return A*np.sin(2*np.pi*f*x+phs) + DC

p0 = [10., .01, 50., np.average(background_mode)]

x = np.arange(0,nimage,1)

popt, pcov = curve_fit(sin, x, background_mode, p0 = p0) # Performing Gaussian fit
	
perr = np.sqrt(np.diag(pcov))

print('##### MODE #####')
print('Optimized sin parameters: %s' %popt)
print('One std deviation error of parameters: %s' %perr)
print('Period of sine wave: %s images' %(1/popt[1]))

plt.figure(1)
plt.scatter(x,background_mode,color='k',marker='.')
plt.plot(x,sin(x,*popt),color='r',label='Sinusoidal Fit')
plt.ylabel('Background Mode')
plt.xlabel('n')
plt.legend()


p0 = [10., .01, 50., np.average(background_avg[100:340])]

x_avg = np.arange(100,340,1)

popt, pcov = curve_fit(sin, x_avg, background_avg[100:340], p0 = p0) # Performing Gaussian fit
	
perr = np.sqrt(np.diag(pcov))

print('\n##### MEAN #####')
print('Optimized sin parameters: %s' %popt)
print('One std deviation error of parameters: %s' %perr)
print('Period of sine wave: %s images' %(1/popt[1]))

plt.figure(2)
plt.scatter(x,background_avg,color='k',marker='.')
plt.plot(x_avg,sin(x_avg,*popt),color='r',label='Sinusoidal Fit')
plt.ylabel('Background Mean')
plt.xlabel('n')
plt.legend()

plt.show()