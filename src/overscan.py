from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

# Open fits file
hdulist = fits.open("../../TrabalhoComputacional/CH_PR100010_TG023201_TU2019-10-06T09-40-30_SCI_RAW_SubArray_V0000.fits")

# Overscan:
overscanleft = hdulist[7].data
(nfigs,ny,nx) = overscanleft.shape

# Perform average of each line (over the 4 columns)
x_avg = np.zeros(200)
for i in range(4):
	x_avg += overscanleft[0,:,i]

x_avg = x_avg/4
print(np.std(x_avg))

x_vals = np.linspace(0,200,200)

plt.figure(1)
plt.scatter(x_vals,x_avg,color='k')
plt.plot(x_vals,[np.average(x_avg) for i in range(len(x_vals))],linestyle = '-.',color = 'r',label = 'Average')
plt.plot(x_vals,[np.average(x_avg) - np.std(x_avg) for i in range(len(x_vals))],color = 'r',label = 'Standard Deviation')
plt.plot(x_vals,[np.average(x_avg) + np.std(x_avg) for i in range(len(x_vals))],color = 'r')
plt.xlabel('Image line')
plt.ylabel('Line Average')
plt.legend()

# Performing different methods to find the average
dif = (np.average(x_avg) - np.average(overscanleft))/np.max(overscanleft)*100
print('Difference between avgs: %s' %dif) # Practically the same so average over the whole array will be used


####### Histogram #######

def gauss(x,*p): # Gaussian curve
	A, mu, sigma = p
	return A*np.exp(-(x-mu)**2/(2.*sigma**2))

hist, bins = np.histogram(x_avg,bins = 50)

bin_centres = ((bins + np.roll(bins,-1))/2)[:-1]

p0 = [10., 5260., 20.]

popt, pcov = curve_fit(gauss, bin_centres, hist, p0 = p0) # Performing Gaussian fit
	
perr = np.sqrt(np.diag(pcov))

print('Optimized gaussian parameters: %s' %popt)
print('One std deviation error of parameters: %s' %perr)

# Plotting

plt.figure(2)
plt.grid(color='gray', linestyle='--', linewidth=0.5)
plt.bar(bins[:-1],hist,align='edge',color='k')
plt.plot(bin_centres,gauss(bin_centres,*popt),color='r',label='Gaussian fit')
plt.xlabel('Averages Values')
plt.ylabel('Counts')
plt.legend()

plt.show()