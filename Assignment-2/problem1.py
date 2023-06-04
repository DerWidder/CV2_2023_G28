from PIL import Image
import numpy as np
np.random.seed(seed=2023)

def load_data(gt_path):
    """Loading the data.
    The same as loading the disparity from Assignment 1 
    (not graded here).

    Args:
        gt_path: path to the disparity image
    
    Returns:
        g_t: numpy array of shape (H, W)
    """
    gt = Image.open(gt_path)
    g_t = np.asarray(gt, dtype=np.float64)

    return g_t

def random_disparity(disparity_size):
    """Compute a random disparity map.

    Args:
        disparity_size: tuple containg height and width (H, W)

    Returns:
        disparity_map: numpy array of shape (H, W)

    """

    disparity_map = np.random.randint(13, size=disparity_size)  # random numbers in [0, 12]

    return disparity_map

def constant_disparity(disparity_size, a):
    """Compute a constant disparity map.

    Args:
        disparity_size: tuple containg height and width (H, W)
        a: the value to initialize with

    Returns:
        disparity_map: numpy array of shape (H, W)

    """
    # reference: https://numpy.org/doc/stable/reference/generated/numpy.full.html
    disparity_map = np.full(disparity_size, a)

    return disparity_map


def log_gaussian(x, mu, sigma):
    """Compute the log gaussian of x.

    Args:
        x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float

    Returns:
        result: numpy array of shape (H, W) (np.float64)

    """
    # 最外围一周点的MRF用什么表示？ 0？
    H, W = np.shape(x)
    result = np.zeros_like(x, dtype=np.float64)  # set float 64 datatype
    '''
     #in the first code, only inside edges are calculated twice, since the boundaries of
     #gt is all zero, the value of P(x) is also doubled.
    
    for i in range(1, H-1):
        for j in range(1, W-1):
            result[i, j] = - ((x[i + 1, j] - x[i, j] - mu)**2 + (x[i, j] - x[i - 1, j] - mu)**2
                              + (x[i, j + 1] - x[i, j] - mu)**2 + (x[i, j] - x[i, j - 1] - mu)**2) / (2 * sigma ** 2)
    '''

    for i in range(H):
        for j in range(W-1):
            result[i, j] = - (x[i, j + 1] - x[i, j] - mu) ** 2 / (2 * sigma ** 2)
    for i in range(H-1):
        for j in range(W):
            result[i, j] += - (x[i + 1, j] - x[i, j] - mu) ** 2 / (2 * sigma ** 2)  # there must add the value

    return result


def mrf_log_prior(x, mu, sigma):
    """Compute the log of the unnormalized MRF prior density.

    Args:
        x: numpy array of shape (H, W) (np.float64)
        mu: float
        sigma: float

    Returns:
        logp: float

    """
    logp = np.sum(log_gaussian(x, mu, sigma))

    return logp

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    gt = load_data('./data/gt.png')

    # Display log prior of GT disparity map
    logp = mrf_log_prior(gt, mu=0, sigma=1.1)
    print("Log Prior of GT disparity map:", logp)

    # Display log prior of random disparity ma
    random_disp = random_disparity(gt.shape)
    logp = mrf_log_prior(random_disp, mu=0, sigma=1.1)
    print("Log-prior of noisy disparity map:",logp)

    # Display log prior of constant disparity map
    constant_disp = constant_disparity(gt.shape, 6)
    logp = mrf_log_prior(constant_disp, mu=0, sigma=1.1)
    print("Log-prior of constant disparity map:", logp)

if __name__ == "__main__":
	main()

