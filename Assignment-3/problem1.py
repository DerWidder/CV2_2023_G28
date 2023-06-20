import math
import gco
import matplotlib.pyplot as plt
import numpy as np
from scipy.sparse import csr_matrix
np.random.seed(seed=2022)

def mrf_denoising_nllh(x, y, sigma_noise):
    """Elementwise negative log likelihood.

      Args:
        x: candidate denoised image
        y: noisy image
        sigma_noise: noise level for Gaussian noise

      Returns:
        A `nd.array` with dtype `float32/float64`.
    """
    nllh = (x - y) ** 2 / (2 * sigma_noise ** 2)
    # nllh = ((x - y) ** 2 / (2 * sigma_noise ** 2)).astype(np.float64)

    assert (nllh.dtype in [np.float32, np.float64])
    return nllh

def edges4connected(height, width):
    """Construct edges for 4-connected neighborhood MRF.
    The output representation is such that output[i] specifies two indices
    of connected nodes in an MRF stored with row-major ordering.

      Args:
        height, width: size of the MRF.

      Returns:
        A `nd.array` with dtype `int32/int64` of size |E| x 2.
    """
    m, n = [], []
    for i in range(height):  # first list all edges in rows, remember to change ordering of nodes in different rows
        for j in range(width - 1):
            m.append(i * width + j)
            n.append(i * width + j + 1)

    for i in range(height - 1):  # then list all edges in columns
        for j in range(width):
            m.append(i * height + j)
            n.append(i * height + j + width)

    edges = np.zeros((len(m), 2)).astype(np.int64)

    for i in range(len(m)):
        edges[i, 0] = m[i]
        edges[i, 1] = n[i]

    assert (edges.shape[0] == 2 * (height*width) - (height+width) and edges.shape[1] == 2)
    assert (edges.dtype in [np.int32, np.int64])
    return edges

def my_sigma():
    return 1

def my_lmbda():
    return 1

def alpha_expansion(noisy, init, edges, candidate_pixel_values, s, lmbda):
    """ Run alpha-expansion algorithm.

      Args:
        noisy: Given noisy grayscale image.
        init: Image for denoising initilisation
        edges: Given neighboor of MRF.
        candidate_pixel_values: Set of labels to consider
        s: sigma for likelihood estimation
        lmbda: Regularization parameter for Potts model.

      Runs through the set of candidates and iteratively expands a label.
      If there have been recorded changes, re-run through the complete set of candidates.
      Stops, if there are no changes in the labelling.

      Returns:
        A `nd.array` of type `int32`. Assigned labels minimizing the costs.
    """
    # reference: http://pmneila.github.io/PyMaxflow/tutorial.html

    H, W = np.shape(noisy)
    pairwise = np.zeros((H*W, H*W))  # create the (H*W, H*W) shape array pairwise
    denoised = np.copy(init)  # create denoised image
    # calculated psnr for the initial state
    psnr_before = compute_psnr(noisy, gt)
    psnr_after = compute_psnr(denoised, gt)

    for i in candidate_pixel_values:  # iterate value from 0 to 254
        unary_r1 = mrf_denoising_nllh(denoised, noisy, s).reshape(1, H * W)  # create first row of unary
        unary_r2 = mrf_denoising_nllh(denoised, i * np.ones_like(noisy), s).reshape(1, H * W)  # create second row of unary
        unary = np.concatenate((unary_r1, unary_r2), axis=0)  # create (2, N) unary array
    
        denoised_pixel_value = denoised.flatten()  # flatten the noisy array
        for j in range(edges.shape[0]):  # update the array pairwise
             index_x, index_y = edges[j]
             if denoised_pixel_value[index_x] != denoised_pixel_value[index_y]:  # Potts model, if the value of two nodes of an edge are not same
                 pairwise[index_x, index_y] = lmbda
    
        graph_labels = gco.graphcut(unary, csr_matrix(pairwise))  # get labels form graphcut
        graph_mask = graph_labels.reshape(H, W)
        denoised[graph_mask == 0] = noisy[graph_mask == 0]  # update the value of denoised
        denoised[graph_mask == 1] = i * np.ones_like(noisy)[graph_mask == 1]
        psnr_after = compute_psnr(denoised, gt)
     print('currently used candidate value: ', i)


    assert (np.equal(denoised.shape, init.shape).all())
    assert (denoised.dtype == init.dtype)
    return denoised

def compute_psnr(img1, img2):
    """Computes PSNR b/w img1 and img2
    para img1: noise image
    para img2: ground truth img
    """
    H, W = np.shape(img1)
    mse = np.sum((img1 - img2) ** 2) / (H * W)

    v_max = np.max(np.concatenate((img1, img2), axis=0))  # we choose the max value of the two images #


    psnr = 10 * np.log10(v_max ** 2 / mse)

    return psnr

def show_images(i0, i1):
    """
    Visualize estimate and ground truth in one Figure.
    Only show the area for valid gt values (>0).
    """

    # Crop images to valid ground truth area
    row, col = np.nonzero(i0)
    plt.figure()
    plt.subplot(1,2,1)
    plt.imshow(i0, "gray", interpolation='nearest')
    plt.subplot(1,2,2)
    plt.imshow(i1, "gray", interpolation='nearest')
    plt.show()

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
if __name__ == '__main__':
    # Read images
    noisy = ((255 * plt.imread('data/la-noisy.png')).squeeze().astype(np.int32)).astype(np.float32)
    gt = (255 * plt.imread('data/la.png')).astype(np.int32)
    
    lmbda = my_lmbda()
    s = my_sigma()

    # Create 4 connected edge neighborhood
    edges = edges4connected(noisy.shape[0], noisy.shape[1])

    # Candidate search range
    labels = np.arange(0, 255)  # return [0, 1, ..., 254]

    # Graph cuts with random initialization
    random_init = np.random.randint(low=0, high=255, size=noisy.shape)  #return np.array shape（H, W）in [0， 255）
    estimated = alpha_expansion(noisy, random_init, edges, labels, s, lmbda)
    psnr_before = compute_psnr(noisy, gt)
    psnr_after = compute_psnr(estimated, gt)
    iter = 1
    while psnr_after <= psnr_before:  # while loop tp get results until the psnr of denosied image is better.
        show_images(noisy, estimated)
        print(psnr_before, psnr_after)
        print(f'results of {iter}-th iteration')
        iter += 1
        estimated = alpha_expansion(noisy, estimated, edges, labels, s, lmbda)
        psnr_after = compute_psnr(estimated, gt)

