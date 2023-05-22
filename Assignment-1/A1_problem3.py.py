
from PIL import Image
import numpy as np
# np.set_printoptions(threshold=np.inf)  # print all the data
np.random.seed(seed=2023)

# convert a RGB image to grayscale
# input (rgb): numpy array of shape (H, W, 3)
# output (gray): numpy array of shape (H, W)
def rgb2gray(rgb):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	gray = 0.1913 * rgb[:, :, 0] + 0.4815 * rgb[:, :, 1] + 0.3273 * rgb[:, :, 2]
	# grayscale weights are given as (0.1913, 0.4815, 0.3272)

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
	i0 = Image.open(i0_path)
	i1 = Image.open(i1_path)
	gt = Image.open(gt_path)
	i_0 = np.asarray(i0, dtype=np.float64)  # convert image values into 64-bit floating np.array
	i_0 = i_0 / 255  # regularization from [0,255] to [0,1]

	i_1 = np.asarray(i1, dtype=np.float64)
	# print(np.max(i_1))
	# print(print(np.max(i_1[:,:,2])))
	i_1 = i_1 / 255  # regularization from [0,255] to [0,1]
	g_t = np.asarray(gt, dtype=np.float64)

	# check if the disparity is (floating point) integer in the range [0,16]
	H, W = np.shape(g_t)
	for i in range(H):
		for j in range(W):
			if g_t[i,j] < 0:
				g_t[i,j]=0
				print('find a value below 0')
			elif g_t[i,j] > 16:
				g_t[i, j] = 16
				print('find a value beyond 16')
			elif (g_t[i,j] % 1) != 0:
				g_t[i,j] = g_t[i,j] // 1
				print('find a non-integer value')


	# print(i_0[:, :, 0])
	# print(g_t)
	# print(np.max((g_t)))
	# print(np.min((g_t)))

	return i_0, i_1, g_t

# image to the size of the non-zero elements of disparity map
# input (img): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (img_crop): numpy array of shape (H', W')
def crop_image(img, d):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	'''
	reference: https://numpy.org/doc/stable/reference/generated/numpy.nonzero.html
	'''
	H_prima, W_prima = np.nonzero(d > 0)  # get the x and y-axis of non-zero values
	h_min = np.min(H_prima)
	h_max = np.max(H_prima)
	w_min = np.min(W_prima)
	w_max = np.max(W_prima)  # get the top left(h_min, w_min) and bottom right(h_max, w_max) points of non-zero value
	# print(h_min)
	# print(h_max)
	# print(w_min)
	# print(w_max)
	img_crop = img[h_min:h_max, w_min:w_max]

	return img_crop

# shift all pixels of i1 by the value of the disparity map
# input (i_1): numpy array of shape (H, W)
# input (d): numpy array of shape (H, W)
# output (i_d): numpy array of shape (H, W)
def shift_disparity(i_1,d):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	H, W = np.shape(i_1)
	i_d = np.zeros((H, W))
	for i in range(H):
		for j in range(W):
			if j - d[i, j] >= 0:
				i_d[i, j] = i_1[i, j - int(d[i, j])]
			else:
				i_d[i, j] = i_1[i, j]  # we need to decide if the disparity is out of boundary

			# i_d[i, j] = i_1[i, j - int(d[i, j])]
	# each pixel of i_d is shifted by the corresponding disparity in d

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
	H, W = np.shape(i_0)
	nll = (1 / (2 * sigma ** 2)) * np.sum(np.square(i_0 - i_1_d - mu)) + \
		  H * W * np.log(np.square(2 * np.pi) * sigma)

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
	H, W = np.shape(i_0)
	nll = (1 / s) * np.sum(np.absolute(i_0 - i_1_d - mu)) + \
		  H * W * np.log(2 * s)

	return nll

# replace p% of the image pixels with values from a normal distribution
# input (img): numpy array of shape (H, W)
# input (p): float
# output (img_noise): numpy array of shape (H, W)
def make_noise(img, p):

	##############################################################################################
	#										IMPLEMENT											 #
	##############################################################################################
	H, W = np.shape(img)
	img_flatten = img.flatten()  # flatten 2 2d-array into 1-d array, in order to use np.random.choice
	# print(np.shape(img_flatten))
	noise = np.random.normal(0.32, 0.78, size=len(img_flatten))  # create noise which is normal distributed
	# should 2-d normal distributed noise be created?
	noise_pixels = np.random.choice(range(len(img_flatten)), size=int(p*H*W/100), replace=False)  # randomly pick p% pixels, no repetition
	# replace the random chosen pixel to the noise
	for i in noise_pixels:
		img_flatten[i] = noise[i]

	img_noise = np.reshape(img_flatten, (H, W))  # reshape the array to be (H, W)

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
	i_1_noise = make_noise(i_1_d, noise)
	gnllh = gaussian_nllh(i_0, i_1_noise, mu, sigma)
	lnllh = laplacian_nllh(i_0, i_1_noise, mu, s)
	print('The negative log of the Gaussian Likelihood: ', gnllh)
	print('The negative log of the Laplacian Likelihood: ', lnllh)

	return gnllh, lnllh

# DO NOT CHANGE
def main():
	# load images
	i0, i1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
	i0, i1 = rgb2gray(i0), rgb2gray(i1)
	#
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
