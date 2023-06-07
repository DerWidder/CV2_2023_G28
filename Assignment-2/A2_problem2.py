import numpy as np

from scipy import interpolate  # Use this for interpolation
from scipy import signal  # Feel free to use convolutions, if needed
from scipy import optimize  # For gradient-based optimisation
from PIL import Image  # For loading images

import matplotlib.pyplot as plt

# for experiments with different initialisation
from problem1 import random_disparity
from problem1 import constant_disparity

np.random.seed(seed=2023)


def rgb2gray(rgb):
    """Converting RGB image to greyscale.
    The same as in Assignment 1 (no graded here).

    Args:
        rgb: numpy array of shape (H, W, 3)

    Returns:
        gray: numpy array of shape (H, W)

    """
    gray = 0.1913 * rgb[:, :, 0] + 0.4815 * rgb[:, :, 1] + 0.3273 * rgb[:, :, 2]
    return gray


def load_data(i0_path, i1_path, gt_path):
    """Loading the data.
    The same as in Assignment 1 (not graded here).

    Args:
        i0_path: path to the first image
        i1_path: path to the second image
        gt_path: path to the disparity image

    Returns:
        i_0: numpy array of shape (H, W)
        i_1: numpy array of shape (H, W)
        g_t: numpy array of shape (H, W)
    """
    i0 = Image.open(i0_path)
    i1 = Image.open(i1_path)
    gt = Image.open(gt_path)
    i_0 = np.asarray(i0, dtype=np.float64)  # convert image into np.array
    i_0 = i_0 / np.max(i_0)
    i_1 = np.asarray(i1, dtype=np.float64)
    i_1 = i_1 / np.max(i_1)
    g_t = np.asarray(gt, dtype=np.float64)

    return i_0, i_1, g_t


def log_gaussian(x, mu, sigma):
    """Calcuate the value and the gradient w.r.t. x of the Gaussian log-density

    Args:
        x: numpy.float 2d-array
        mu and sigma: scalar parameters of the Gaussian

    Returns:
        value: value of the log-density
        grad: gradient of the log-density w.r.t. x
    """
    # return the value and the gradient
    H, W = np.shape(x)
    result = np.zeros_like(x, dtype=np.float64)  # set float 64 datatype
    grad = np.zeros_like(x, dtype=np.float64)
    # for i in range(1, H - 1):
    #     for j in range(1, W - 1):
    #         result[i, j] = - ((x[i, j] - x[i + 1, j] - mu) ** 2 + (x[i, j] - x[i - 1, j] - mu) ** 2
    #                           + (x[i, j] - x[i, j + 1] - mu) ** 2 + (x[i, j] - x[i, j - 1] - mu) ** 2) / (
    #                                    2 * sigma ** 2)
    for i in range(H):
        for j in range(W - 1):
            result[i, j] = - (x[i, j + 1] - x[i, j] - mu) ** 2 / (2 * sigma ** 2)
    for i in range(H - 1):
        for j in range(W):
            result[i, j] += - (x[i + 1, j] - x[i, j] - mu) ** 2 / (2 * sigma ** 2)  # there must add the value

    for i in range(1, H - 1):
        for j in range(1, W - 1):
            grad[i, j] = - (4 * x[i, j] - x[i + 1, j] - x[i - 1, j] - x[i, j + 1] - x[i, j - 1]) / sigma ** 2

    value = np.sum(result)

    return value, grad


def stereo_log_prior(x, mu, sigma):
    """Evaluate gradient of pairwise MRF log prior with Gaussian distribution

    Args:
        x: numpy.float 2d-array (disparity)

    Returns:
        value: value of the log-prior
        grad: gradient of the log-prior w.r.t. x
    """
    value, grad = log_gaussian(x, mu, sigma)
    # prior, disparity
    return value, grad


def shift_interpolated_disparity(im1, d):
    """Shift image im1 by the disparity value d.
    Since disparity can now be continuous, use interpolation.

    Args:
        im1: numpy.float 2d-array  input image
        d: numpy.float 2d-array  disparity

    Returns:
        im1_shifted: Shifted version of im1 by the disparity value.
    """
    H, W = np.shape(im1)
    shifted_im1 = np.zeros_like(im1)
    points = np.linspace(0, W, num=W, endpoint=False)  # create an array [0, 1, 2, ..., W-1]
    for i in range(H):
        # reference https://docs.scipy.org/doc/scipy/reference/interpolate.html
        # f_inter = interpolate.interp1d(points - d[i, :], im1[i, :], kind='cubic')  # create a cubic interpolate function by using all
        #                                                                  # values of each row
        # shifted_im1[i, :] = f_inter(points)
        f_inter = interpolate.interp1d(points, im1[i, :],
                                       kind='cubic')  # create a cubic interpolate function by using all
        # values of each row
        # attention, the interpolated points should locate inside the boundary [0, W-1]
        points_shifted = points - d[i, :]
        # for j in range(np.size(points)):  # method 1
        #     if points_shifted[j] < 0:
        #         points_shifted[j] = 0
        #     if points_shifted[j] > W - 1:
        #         points_shifted[j] = W - 1
        #

        # for index, value in enumerate(points_shifted):  # method 2
        #     if value < 0:
        #         points_shifted[index] = 0
        #     if value > W - 1:
        #         points_shifted[index] = W - 1

        points_shifted[points_shifted < 0] = 0
        points_shifted[points_shifted > (W - 1)] = W - 1

        shifted_im1[i, :] = f_inter(points_shifted)

    return shifted_im1


def stereo_log_likelihood(x, im0, im1, mu, sigma):
    """Evaluate gradient of the log likelihood.

    Args:
        x: numpy.float 2d-array of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        value: value of the log-likelihood
        grad: gradient of the log-likelihood w.r.t. x

    Hint: Make use of shift_interpolated_disparity and log_gaussian
    """
    im1_shifted = shift_interpolated_disparity(im1, x)
    value, grad = log_gaussian(im0 - im1_shifted, mu, sigma)
    # likelihood, pixel intensity
    return value, grad


def stereo_log_posterior(d, im0, im1, mu, sigma, alpha):
    """Computes the value and the gradient of the log-posterior

    Args:
        d: numpy.float 2d-array of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        value: value of the log-posterior
        grad: gradient of the log-posterior w.r.t. x
    """
    # log_posterior, log_posterior_grad = stereo_log_likelihood(d, im0, im1, mu, sigma) + alpha * log_gaussian(d, mu, sigma)
    val_likelihood, grad_likelihood = stereo_log_likelihood(d, im0, im1, mu, sigma)
    val_prior, grad_prior = log_gaussian(d, mu, sigma)
    log_posterior = val_likelihood + alpha * val_prior
    log_posterior_grad = grad_likelihood + alpha * grad_prior
    print(log_posterior)
    return log_posterior, log_posterior_grad


def optim_method():
    """Simply returns the name (string) of the method
    accepted by scipy.optimize.minimize, that you found
    to work well.
    This is graded with 1 point unless the choice is arbitrary/poor.
    """
    # return 'BFGS'
    # return 'SLSQP'
    return 'CG'
    # return 'Newton-CG'


def stereo(d0, im0, im1, mu, sigma, alpha, method=optim_method()):
    """Estimating the disparity map

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        d: numpy.float 2d-array estimated value of the disparity
    """
    H, W = np.shape(d0)

    def fun(args):
        im0, im1, mu, sigma, alpha = args
        # def v(x):
        #     return -stereo_log_posterior(x.reshape((H, W)), im0, im1, mu, sigma, alpha)[0]
        v = lambda x: -stereo_log_posterior(x.reshape((H, W)), im0, im1, mu, sigma, alpha)[0]
        return v

    def grad(args):
        im0, im1, mu, sigma, alpha = args
        # def v(x):
        #     return -stereo_log_posterior(x.reshape((H, W)), im0, im1, mu, sigma, alpha)[0]
        jac = lambda x: -stereo_log_posterior(x.reshape((H, W)), im0, im1, mu, sigma, alpha)[1].flatten()
        return jac

    args = (im0, im1, mu, sigma, alpha)
    x0 = d0.flatten()
    # print(fun(args)(x0))
    # print(grad(args)(x0))
    res = optimize.minimize(fun(args), x0, method=method, jac=grad(args), tol=1e0)
    # res = optimize.minimize(lambda x: (x-1) ** 2,np.asarray(4),method='SLSQP')
    print(res.success)
    # print(res.message)
    d0 = res.x.reshape((H, W))
    return d0


'''
copy from CV1 Assignment-2
'''


def gaussian_kernel(fsize, sigma):
    '''
    Define a Gaussian kernel

    Args:
        fsize: kernel size
        sigma: sigma of Gaussian kernel

    Returns:
        The Gaussian kernel
    '''

    kernel = np.fromfunction(lambda x, y: (1 / (2 * np.pi * sigma ** 2)) * np.exp(
        (-1 * ((x - (fsize - 1) / 2) ** 2 + (y - (fsize - 1) / 2) ** 2)) / (2 * sigma ** 2)), (fsize, fsize))
    return kernel


'''
copy from CV1 Assignment-2
'''


def upsample_x2(x, factor=2):
    '''
    Upsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        Upsampled image as numpy array (2*H, 2*W) with interpolation

    '''

    """ 
        reference: https://stackoverflow.com/questions/37662180/interpolate-missing-values-2d-python
    """
    # output every two pixels, x2 = x1[::factor] is same as x2 = x1[0: x1.size: factor]  slicing in python

    H, W = np.shape(x)
    # print('shape of original:', np.shape(x))
    a = np.zeros((2 * H, 2 * W))
    a[::factor, ::factor] = x

    ## if no further codes, the upsampled pixels are all zero

    a[a == 0] = np.nan  # set all the upsampled pixels to be np.nan, in order to get a mask
    m = np.arange(0, a.shape[1])
    n = np.arange(0, a.shape[0])
    array = np.ma.masked_invalid(a)  # mask the pixels whose value is np.nan
    xx, yy = np.meshgrid(m, n)
    x1 = xx[~array.mask]  # get only the valid values
    y1 = yy[~array.mask]
    newarr = array[~array.mask]

    upsample = interpolate.griddata((x1, y1), newarr.ravel(), (xx, yy), method='cubic',
                                    fill_value=0)  # the boundaries are set to be zero
    # print('shape of upsampling:', np.shape(upsample))
    return upsample


'''
copy from CV1 Assignment-2
'''


def downsample_x2(x, factor=2):
    '''
    Downsampling an image by a factor of 2

    Args:
        x: image as numpy array (H * W)

    Returns:
        downsampled image as numpy array (H/2 * W/2)

    '''

    """ Attention: H/2 in Python 3 is float we can use H//2 to change into int, same as int(H/2) 
        reference: https://stackoverflow.com/questions/34231244/downsampling-a-2d-numpy-array-in-python/34232507
    """
    # output every two pixels, x2 = x1[::factor] is same as x2 = x1[0: x1.size: factor]  slicing in python
    downsample = x[::factor, ::factor]

    return downsample


'''
copy from CV1 Assignment-2
'''


def gaussian_pyramid(img, nlevels, fsize, sigma):
    '''
    A Gaussian pyramid is constructed by combining a Gaussian kernel and downsampling.
    Tips: use scipy.signal.convolve2d for filtering image.

    Args:
        img: face image as numpy array (H * W)
        nlevels: number of levels of Gaussian pyramid, in this assignment we will use 3 levels
        fsize: Gaussian kernel size, in this assignment we will define 5
        sigma: sigma of Gaussian kernel, in this assignment we will define 1.4

    Returns:
        GP: list of Gaussian downsampled images, it should be 3 * H * W
    '''

    #  use convolve2d   first
    GP = []
    kernel = gaussian_kernel(fsize, sigma)

    for i in range(nlevels):
        if i == 0:
            GP.append(img)  # first should be the original image
            print('shape of original img:', np.shape(img))
            # GP.append(convolve2d(img, kernel, boundary='fill')) # first image without downsampling
        else:
            # temp = convolve2d(downsample_x2(GP[i-1]), kernel, boundary='fill') # first downsampling and then filtering
            temp = downsample_x2(
                signal.convolve2d(GP[i - 1], kernel, mode='same'))  # first filtering and then downsampling
            print('shape of downsampling:', np.shape(temp))
            GP.append(temp)

    return GP


def coarse2fine(d0, im0, im1, mu, sigma, alpha, num_levels):
    """Coarse-to-fine estimation strategy. Basic idea:
        1. create an image pyramid (of size num_levels)
        2. starting with the lowest resolution, estimate disparity
        3. proceed to the next resolution using the estimated
        disparity from the previous level as initialisation

    Args:
        d0: numpy.float 2d-array initialisation of the disparity
        im0: numpy.float 2d-array of image #0
        im1: numpy.float 2d-array of image #1

    Returns:
        pyramid: a list of size num_levels containing the estimated
        disparities at each level (from finest to coarsest)
        Sanity check: pyramid[0] contains the finest level (highest resolution)
                      pyramid[-1] contains the coarsest level
    """

    GP_im0 = gaussian_pyramid(im0, num_levels, 5, 1.4)  # parameters are chosen the same as in CV1 Assignment-2
    GP_im1 = gaussian_pyramid(im1, num_levels, 5, 1.4)
    GP_d = list(np.zeros((num_levels, 1)))
    d0_init = gaussian_pyramid(d0, num_levels, 5, 1.4)[-1]  # the initialisation of  coarsest disparity

    for i in range(num_levels):
        if i == 0:
            GP_d[-1] = stereo(d0_init, GP_im0[-1], GP_im1[-1], mu, sigma, alpha, method=optim_method())
        else:
            GP_d[num_levels - 1 - i] = stereo(upsample_x2(GP_d[num_levels - i]), GP_im0[num_levels - 1 - i],
                                              GP_im1[num_levels - 1 - i], mu, sigma, alpha, method=optim_method())

    return GP_d


# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():
    # these are the same functions from Assignment 1
    # (no graded in this assignment)
    im0, im1, gt = load_data('./data/i0.png', './data/i1.png', './data/gt.png')
    im0, im1 = rgb2gray(im0), rgb2gray(im1)

    mu = 0.1
    sigma = 1.7

    # experiment with other values of alpha
    alpha = [0.1, 0.25, 0.5, 0.75, 1]
    # alpha = 1.0

    # initial disparity map
    # experiment with constant/random values
    # d0 = gt
    # d0 = random_disparity(gt.shape)
    # d0 = constant_disparity(gt.shape, 6)
    d0 = [gt, constant_disparity(gt.shape, 6), random_disparity(gt.shape)]
    label = ['ground truth', 'constant value', 'random noise']

    # visualize ground true
    # disparity=stereo(d0[1], im0, im1, mu, sigma, alpha=1)
    # fig = plt.figure()
    # ax1 = fig.add_subplot(1, 2, 1)
    # ax1.imshow(gt)
    # ax2 = fig.add_subplot(1, 2, 2)
    # ax2.imshow(disparity)
    # plt.show()

    disparity_list = []
    for i in range(3):
        # get the disparity of the given image pair
        disparity_list.append(stereo(d0[i], im0, im1, mu, sigma, alpha=0.1))
    # visualize the result
    fig = plt.figure()
    ax1 = fig.add_subplot(2, 2, 1)
    ax1.imshow(gt)
    ax2 = fig.add_subplot(2, 2, 2)
    ax2.imshow(disparity_list[0])
    ax3 = fig.add_subplot(2, 2, 3)
    ax3.imshow(disparity_list[1])
    ax4 = fig.add_subplot(2, 2, 4)
    ax4.imshow(disparity_list[2])
    ax1.axes.xaxis.set_visible(False)
    ax1.axes.yaxis.set_visible(False)
    ax2.axes.xaxis.set_visible(False)
    ax2.axes.yaxis.set_visible(False)
    ax3.axes.xaxis.set_visible(False)
    ax3.axes.yaxis.set_visible(False)
    ax4.axes.xaxis.set_visible(False)
    ax4.axes.yaxis.set_visible(False)
    plt.show()


    for idx in range(len(alpha)):
        print('case', idx+1, ': alpha =', alpha[idx])
        for i in range(3):
            # get the disparity of the given image pair
            disparity = stereo(d0[i], im0, im1, mu, sigma, alpha[idx])
            # visualize the std between result and gt
            std = np.std(disparity-gt)
            print('STD between gt and',label[i] ,' is ',std)

    # Display stereo: Initialized with noise
    # test_gt = Image.fromarray(d0)
    # test_gt.show(title='before')
    # test_image = Image.fromarray(disparity)
    # test_image.show(title='after')

    # Pyramid
    num_levels = 3
    # pyramid = coarse2fine(d0, im0, im1, mu, sigma, alpha, num_levels)


if __name__ == "__main__":
    main()