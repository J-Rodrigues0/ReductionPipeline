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

	print("--- IMAGE REDUCTION PIPELINE ---")

	t0 = clock()

	#### CHOOSE FITS and FF FILE PATHS ####

	fits_path = "../../TrabalhoComputacional/CH_PR100010_TG023201_TU2019-10-06T09-40-30_SCI_RAW_SubArray_V0000.fits"
	flat_path = "../../TrabalhoComputacional/CH_PR100010_TG023201_TU2019-10-06T09-40-00_SIM_TRU_FlatField_V0000.fits"

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
		images_b = overscan(hdulist,images)
		if clocking:
			print("ELAPSED TIME AFTER OVERSCAN: %s ms" %int((clock()-t0)*1000))
	except:
		print("ERROR DURING OVERSCAN")
		raise

	#### FLATFIELD ####

	try:
		images_f = flatfield(hdulist,images_b,flat_path)
		if clocking:
			print("ELAPSED TIME AFTER FLATFIELD: %s ms" %int((clock()-t0)*1000))
	except:
		print("ERROR DURING FLATFIELD")
		raise
		
	#### CENTER OF MASS CALCULATION ####

	try:
		R = calc_CM(images_f)
		if clocking:
			print("ELAPSED TIME AFTER CENTER OF MASS CALCULATION: %s ms" %int((clock()-t0)*1000))
	except:
		print("ERROR IN CENTER OF MASS CALCULATION")
		raise

	#### TARGET DETECTION ####

	try:
		target = calc_TARGET(images_f,R)
		if clocking:
			print("ELAPSED TIME AFTER TARGET DETECTION: %s ms" %int((clock()-t0)*1000))
	except:
		print("ERROR IN TARGET DETECTION")
		raise

	#### CALCULATE BACKGROUND ####

	try:
		bkg_mode, bkg_mean = calc_background(images_f,R)
		if clocking:
			print("ELAPSED TIME AFTER BACKGROUND CALCULATION: %s ms" %int((clock()-t0)*1000))
	except:
		print("ERROR IN BACKGROUND CALCULATION")
		raise

	#### PHOTOMETRY ####

	try:
		flux = photometry(images_f,R,target,bkg_mode=bkg_mode,bkg_mean=bkg_mean)
		if clocking:
			print("ELAPSED TIME AFTER PHOTOMETRY: %s ms" %int((clock()-t0)*1000))
	except:
		print("ERROR IN PHOTOMETRY")
		raise

	#### DATA TREATMENT ####

	print("--- TOTAL ELAPSED TIME: %s ms ---" %int((clock()-t0)*1000))
	return 0

if __name__ == "__main__":
	tk = Tk()
	fits_path = askopenfilename(initialdir = os.getcwd(),title = "Select SUBARRAY file")
	flat_path = askopenfilename(initialdir = os.getcwd(),title = "Select FLATFIELD file")

	clocking = messagebox.askyesno("Clocking","Print clocking information?")
	output_path = askdirectory(initialdir = os.getcwd(),title = "Select OUTPUT directory")
	tk.destroy()

	main(clocking = True)


