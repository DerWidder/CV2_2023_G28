
from PIL import Image
import numpy as np
np.random.seed(seed=2023)

# convert a RGB image to grayscale
# input (rgb): numpy array of shape (H, W, 3)
# output (gray): numpy array of shape (H, W)
def rgb2gray(rgb):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

	return gray

#load the data
# input (i0_path): path to the first image
# input (i1_path): path to the second image
# input (gt_path): path to the disparity image
# output (i_0): numpy array of shape (H, W, 3)
# output (i_1): numpy array of shape (H, W, 3)
# output (g_t): numpy array of shape (H, W)
def load_data(i0_path, i1_path, gt_path):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

	return i_0, i_1, g_t

# image to the size of the non-zero elements of disparity map
# input (img): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (img_crop): numpy array of shape (H', W')
def crop_image(img, d):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

	return img_crop

# shift all pixels of i1 by the value of the disparity map
# input (i_1): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (i_d): numpy array of shape (H, W)
def shift_disparity(i_1,d):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

	return i_d

# compute the negative log of the Gaussian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (sigma): float
# output (nll): numpy scalar of shape ()
def gaussian_nllh(i_0, i_1_d, mu, sigma):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

	return nll

# compute the negative log of the Laplacian likelihood
# input (i_0): numpy array of shape (H, W)
# input (i_1_d): numpy array of shape (H, W)
# input (mu): float
# input (s): float
# output (nll): numpy scalar of shape ()
def laplacian_nllh(i_0, i_1_d, mu,s):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

	return nll

# replace p% of the image pixels with values from a normal distribution
# input (img): numpy array of shape (H, W)
# input (p): float
# output (img_noise): numpy array of shape (H, W)
def make_noise(img, p):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

	return img_noise

# apply noise to i1_sh and return the values of the negative lok-likelihood for both likelihood models with mu, sigma, and s
# input (i0): numpy array of shape (H, W)
# input (i1_sh): numpy array of shape (H, W)
# input (noise): float
# input (mu): float
# input (sigma): float
# input (s): float
# output (gnllh) - gaussian negative log-likelihood: numpy scalar of shape ()
# output (lnllh) - laplacian negative log-likelihood: numpy scalar of shape ()
def get_nllh_for_corrupted(i_0, i_1_d, noise, mu, sigma, s):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################

	return gnllh, lnllh

# DO NOT CHANGE
def main():
	# load images
	i0, i1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
	i0, i1 = rgb2gray(i0), rgb2gray(i1)

	# shift i1
	i1_sh = shift_disparity(i1, gt)

	# crop images
	i0 = crop_image(i0, gt)
	i1_sh = crop_image(i1_sh, gt)

	mu = 0.0
	sigma = 1.3
	s = 1.3
	for noise in [0.0, 15.0, 28.0]:

		gnllh, lnllh = get_nllh_for_corrupted(i0, i1_sh, noise, mu, sigma, s)

if __name__ == "__main__":
	main()
