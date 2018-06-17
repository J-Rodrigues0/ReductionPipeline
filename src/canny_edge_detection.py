from scipy import ndimage
from scipy.ndimage import sobel, generic_gradient_magnitude, generic_filter
from scipy.ndimage.filters import gaussian_filter
import numpy as np
import matplotlib.pyplot as plt

def round_angle(angle):
    """
	Rounds angles to find neighbours.
    """
    angle = np.rad2deg(angle) % 180
    if (0 <= angle < 22.5) or (157.5 <= angle < 180):
        angle = 0
    elif (22.5 <= angle < 67.5):
        angle = 45
    elif (67.5 <= angle < 112.5):
        angle = 90
    elif (112.5 <= angle < 157.5):
        angle = 135
    return angle

def gs_filter(img, sigma):
	"""
	Applies gaussian filter.
	"""
	return gaussian_filter(img, sigma)

def grad_intensity(img):
	"""
	Computes gradient intensity of the images.
	"""

	Kx = np.array(
		[[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], np.int32
	)
	# Kernel for Gradient in y-direction
	Ky = np.array(
		[[1, 2, 1], [0, 0, 0], [-1, -2, -1]], np.int32
	)
	# Apply kernels to the image
	Ix = ndimage.filters.convolve(img, Kx)
	Iy = ndimage.filters.convolve(img, Ky)

	# return the hypothenuse of (Ix, Iy)
	G = np.hypot(Ix, Iy)
	D = np.arctan2(Iy, Ix)

	return (G, D)

def threshold(img, t, T):
	""" Thresholding
	Iterates through image pixels and marks them as WEAK and STRONG edge
	pixels based on the threshold values.

	t: lower threshold
	T: upper threshold
	"""

	# define gray value of a WEAK and a STRONG pixel
	cf = {
	    'WEAK': np.int32(50),
	    'STRONG': np.int32(255),
	}

	# get strong pixel indices
	strong_i, strong_j = np.where(img > T)

	# get weak pixel indices
	weak_i, weak_j = np.where((img >= t) & (img <= T))

	# get pixel indices set to be zero
	zero_i, zero_j = np.where(img < t)

	# set values
	img[strong_i, strong_j] = cf.get('STRONG')
	img[weak_i, weak_j] = cf.get('WEAK')
	img[zero_i, zero_j] = np.int32(0)

	return (img, cf.get('WEAK'))

def tracking(img, weak, strong=255):
	"""
	Checks if edges marked as weak are connected to strong edges.

	img: image to be processed (thresholded image)
	weak: Value that was used to mark a weak edge in Step 4
	"""

	M, N = img.shape
	for i in range(M):
		for j in range(N):
			if img[i, j] == weak:
				# check if one of the neighbours is strong (=255 by default)
				try:
					if ((img[i + 1, j] == strong) or (img[i - 1, j] == strong)
						or (img[i, j + 1] == strong) or (img[i, j - 1] == strong)
						or (img[i+1, j + 1] == strong) or (img[i-1, j - 1] == strong)):
						img[i, j] = strong
					else:
						img[i, j] = 0
				except IndexError as e:
					pass
	return img

def canny_detect(img,R,sigma,t,T,square_size = 50):
	"""
	Applies the several steps of the algorithm.
	"""

	img_square = img[(R[1] - square_size):(R[1] + square_size),(R[0] - square_size):(R[0] + square_size)].copy()

	img_square = gs_filter(img_square,sigma)

	G,D = grad_intensity(img_square)

	img_square, weak = threshold(img_square,t,T)

	img_square = tracking(img_square,weak,strong=255)

	indexes = np.where(img_square == 255)

	return np.array([indexes[0] + int(img.shape[0]/2)-square_size, indexes[1] + int(img.shape[0]/2)-square_size])
