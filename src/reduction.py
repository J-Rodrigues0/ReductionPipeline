"""

FINAL IMAGE REDUCTION PIPELINE

- Implements the algorithm in reduction_demonstration.py
- Timing optimized

"""

#### IMPORTING RELEVANT LIBRARIES ####

from astropy.io import fits
import matplotlib.pyplot as plt
import numpy as np
from time import clock
from reduction_inc import *
from tkinter import messagebox,Tk
from tkinter.filedialog import askopenfilename, askdirectory
import os

def main(**kwargs):
	clocking = kwargs['clocking']
	out_path = kwargs['out_path']
	
	#### FITS AND FLAT PATHS ####

	fits_path = kwargs['fits_path']
	flat_path = kwargs['flat_path']

	print("--- IMAGE REDUCTION PIPELINE ---")

	t0 = clock()

	#### OPEN FITS ####

	try:
		hdulist = fits.open(fits_path);
		print("FITS SUCCESSFULLY OPENED")
	except FileNotFoundError:
		print("FITS FILE PATH IS INCORRECT")
		return

	images = hdulist[1].data

	#### OVERSCAN ####

	try:
		t = clock()
		images_b = overscan(hdulist,images)
		if clocking:
			print("ELAPSED TIME IN OVERSCAN: %s ms" %int((t-t0)*1000))
	except:
		print("ERROR DURING OVERSCAN")
		raise

	#### FLATFIELD ####

	try:
		t = clock()
		images_f = flatfield(hdulist,images_b,flat_path)
		if clocking:
			print("ELAPSED TIME IN FLATFIELD: %s ms" %int((t-t0)*1000))
	except:
		print("ERROR DURING FLATFIELD")
		raise
		
	#### CENTER OF MASS CALCULATION ####

	try:
		t = clock()
		R = calc_CM(images_f)
		if clocking:
			print("ELAPSED TIME IN CENTER OF MASS CALCULATION: %s ms" %int((t-t0)*1000))
	except:
		print("ERROR IN CENTER OF MASS CALCULATION")
		raise

	#### TARGET DETECTION ####

	try:
		t = clock()
		target = calc_TARGET(images_f,R)
		if clocking:
			print("ELAPSED TIME IN TARGET DETECTION: %s ms" %int((t-t0)*1000))
	except:
		print("ERROR IN TARGET DETECTION")
		raise

	#### CALCULATE BACKGROUND ####

	try:
		t = clock()
		bkg_mode, bkg_mean = calc_background(images_f,R)
		if clocking:
			print("ELAPSED TIME IN BACKGROUND CALCULATION: %s ms" %int((t-t0)*1000))
	except:
		print("ERROR IN BACKGROUND CALCULATION")
		raise

	#### PHOTOMETRY ####

	try:
		t = clock()
		kwargs = {}
		#kwargs['bkg_mean'] = bkg_mean # Background mean is optional. See photometry() for more info.
		flux = photometry(images_f,R,target,bkg_mode,**kwargs)
		if clocking:
			print("ELAPSED TIME IN PHOTOMETRY: %s ms" %int((t-t0)*1000))
	except:
		print("ERROR IN PHOTOMETRY")
		raise

	#### DATA TREATMENT ####

	try:
		t = clock()
		fits_name = fits_path.split('/')[-1]
		flat_name = fits_path.split('/')[-1]

		data = np.zeros([len(flux),3])
		data[:,0] = flux
		data[:,1] = bkg_mode
		data[:,2] = bkg_mean

		kwargs = {}
		kwargs['fits_name'] = fits_name
		kwargs['flat_name'] = flat_name
		kwargs['data'] = data

		# Background mean and mode plotting is optional. See photometry() for more info.
		kwargs['bkg_mean'] = bkg_mean
		kwargs['bkg_mode'] = bkg_mode

		treat_data(flux,out_path,**kwargs)
		if clocking:
			print("ELAPSED TIME IN DATA TREATMENT: %s ms" %int((t-t0)*1000))
	except:
		print("ERROR IN DATA TREATMENT")
		raise

	print("--- TOTAL ELAPSED TIME: %s ms ---" %int((clock()-t0)*1000))
	return 0

if __name__ == "__main__":

	tk = Tk()
	fits_path = askopenfilename(initialdir = os.getcwd(),title = "Select SUBARRAY file")
	flat_path = askopenfilename(initialdir = os.getcwd(),title = "Select FLATFIELD file")

	out_path = askdirectory(initialdir = os.getcwd(),title = "Select OUTPUT directory")
	clocking = messagebox.askyesno("Clocking","Print clocking information?")
	tk.destroy()

	kwargs = {}
	kwargs['clocking'] = clocking
	kwargs['fits_path'] = fits_path
	kwargs['flat_path'] = flat_path
	kwargs['out_path'] = out_path

	main(**kwargs)



