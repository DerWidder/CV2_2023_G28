from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as tf
import torch.optim as optim

from utils import flow2rgb
from utils import rgb2gray
from utils import read_flo
from utils import read_image

np.random.seed(seed=2022)


def numpy2torch(array):
    """ Converts 3D numpy (H,W,C) ndarray to 3D PyTorch (C,H,W) tensor.

    Args:
        array: numpy array of shape (H, W, C)
    
    Returns:
        tensor: torch tensor of shape (C, H, W)
    """
    # tensor = torch.from_numpy(array)

    # H, W, C = np.shape(array)
    # tensor = torch.zeros((C, H, W))
    # for i in range(C):
    #     tensor[i, :, :] = array[:, :, i]
    tensor = torch.from_numpy(np.transpose(array, (2, 0, 1)))  # 维度映射，将原来的第三维度（2）映射到第一维度（0）

    return tensor


def torch2numpy(tensor):
    """ Converts 3D PyTorch (C,H,W) tensor to 3D numpy (H,W,C) ndarray.

    Args:
        tensor: torch tensor of shape (C, H, W)
    
    Returns:
        array: numpy array of shape (H, W, C)
    """
    # array = tensor.numpy()

    # C, H, W = tensor.size()
    # array = np.zeros(H, W, C)
    # for i in range(C):
    #     array[:, :, i] = tensor[i, :, :]
    array = np.transpose(tensor.numpy(), (1, 2, 0))

    return array


def load_data(im1_filename, im2_filename, flo_filename):
    """Loading the data. Returns 4D tensors. You may want to use the provided helper functions.

    Args:
        im1_filename: path to image 1
        im2_filename: path to image 2
        flo_filename: path to the ground truth flow
    
    Returns:
        tensor1: torch tensor of shape (B, C, H, W)
        tensor2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    """
    im1_gray = rgb2gray(read_image(im1_filename))
    im2_gray = rgb2gray(read_image(im2_filename))
    gt = read_flo(flo_filename)

    im1_tensor_3d = numpy2torch(im1_gray)
    tensor1 = im1_tensor_3d.unsqueeze(0)  # unsqueeze()用于增加一个维度
    im2_tensor2_3d = numpy2torch(im2_gray)
    tensor2 = im2_tensor2_3d.unsqueeze(0)
    flow_gt_tensor_3d = numpy2torch(gt)
    flow_gt = flow_gt_tensor_3d.unsqueeze(0)

    return tensor1, tensor2, flow_gt


def evaluate_flow(flow, flow_gt):
    """Evaluate the average endpoint error w.r.t the ground truth flow_gt.
    Excludes pixels, where u or v components of flow_gt have values > 1e9.

    Args:
        flow: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:
        aepe: torch tensor scalar 
    """
    B, C, H, W = flow.shape
    EPE = torch.zeros(H, W)
    # flow_gt[flow_gt > 1e9] = 0
    # flow[flow_gt > 1e9] = 0
    #
    # for i in range(H):
    #     for j in range(W):
    #         for m in range(C):
    #             EPE[i, j] += torch.pow((flow[:, m, i, j] - flow_gt[:, m, i, j]), 2)
    #         EPE[i, j] = torch.sqrt(EPE[i, j])
    for i in range(H):
        for j in range(W):
            if flow_gt[:, 0, i, j] > 1e9:
                EPE[i, j] = 0
            else:
                EPE[i, j] = torch.sqrt(torch.pow((flow[:, 0, i, j] - flow_gt[:, 0, i, j]), 2) +
                                       torch.pow((flow[:, 1, i, j] - flow_gt[:, 1, i, j]), 2))

    aepe = torch.sum(EPE) / (H*W)

    return aepe


def visualize_warping_practice(im1, im2, flow_gt):
    """ Visualizes the result of warping the second image by ground truth.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
    
    Returns:

    """
    im2_warp = warp_image(im2, flow_gt)  # warp the second image by using the gt
    diff = torch.abs(im1 - im2_warp)  # calculate the difference between im1 and im2_warp
    # why abs

    # convert the tensor back to ndarray
    im1 = np.squeeze(im1, axis=0)  # squeeze()消除一维
    im1_np = torch2numpy(im1)
    im2_warp = np.squeeze(im2_warp, axis=0)
    im2_warp_np = torch2numpy(im2_warp)
    diff = np.squeeze(diff, axis=0)
    diff_np = torch2numpy(diff)

    # display the im1, im2_warp, diff in subplots
    fig, axs = plt.subplots(1, 3, figsize=(12, 4))
    axs[0].imshow(im1_np[:, :, 0], cmap='gray')
    axs[0].set_title('im1')

    axs[1].imshow(im2_warp_np[:, :, 0], cmap='gray')
    axs[1].set_title('im2_warp')

    axs[2].imshow(diff_np[:, :, 0], cmap='gray')
    axs[2].set_title('difference')

    for ax in axs:
        ax.axis('off')
    plt.tight_layout()
    plt.show()

    return


def warp_image(im, flow):
    """ Warps given image according to the given optical flow.

    Args:
        im: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
    
    Returns:
        x_warp: torch tensor of shape (B, C, H, W)
    """
    # reference: https://discuss.pytorch.org/t/how-to-warp-the-image-with-optical-flow-and-grid-sample/71531
    # input(N, C, H_in, W_in), grid(N, H_out, W_out, 2)
    # B, C, H, W = im.size()
    #
    # # build mesh grid
    # xx = torch.arange(0, W).view(1, -1).repeat(H, 1)
    # yy = torch.arange(0, H).view(-1, 1).repeat(1, W)
    # xx = xx.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # yy = yy.view(1, 1, H, W).repeat(B, 1, 1, 1)
    # grid = torch.cat((xx, yy), 1).float()
    #
    # im = im.cuda()
    # grid = grid.cuda()
    # vgrid = torch.autograd.Variable(grid) + flow
    #
    # # scale grid to [-1,1]
    # vgrid[:, 0, :, :] = 2.0 * vgrid[:, 0, :, :].clone() / max(W - 1, 1) - 1.0
    # vgrid[:, 1, :, :] = 2.0 * vgrid[:, 1, :, :].clone() / max(H - 1, 1) - 1.0
    #
    # vgrid = vgrid.permute(0, 2, 3, 1)  # from B,2,H,W -> B,H,W,2，为什么要这么变呢？是因为要配合grid_sample这个函数的使用
    # output = tf.grid_sample(im, vgrid)
    # mask = torch.autograd.Variable(torch.ones(im.size())).cuda()
    # mask = tf.grid_sample(mask, vgrid)
    #
    # mask[mask < 0.9999] = 0
    # mask[mask > 0] = 1
    #
    # return output*mask

    # input(N, C, H_in, W_in), grid(N, H_out, W_out, 2)
    B, C, H, W = im.shape

    # build mesh grid
    grid_y, grid_x = torch.meshgrid(torch.arange(H), torch.arange(W))
    grid = torch.stack((grid_x, grid_y), dim=0).float()  # torch.stack 对输入的张量进行连接，新增一个维度

    # add the flow to grid
    warped_grid = grid + flow  # 相当于x_ij + u_ij, y_ij + v_ij

    # Normalize grid to [-1, 1]
    grid_normalized = (warped_grid / torch.tensor([W - 1, H - 1]).view(2, 1, 1)) * 2 - 1
    # torch.Tensor.view()相当于numpy.reshape()

    # Reshape normalized grid
    grid_normalized = grid_normalized.permute(0, 2, 3, 1)
    # parameter grid in function grid_sample should be (N, H_out, W_out, 2) in 4-D case.
    # torch.transpose一次只能换两个维度， torch.permute可以进行多维度变换

    # Warp the image using grid sample
    x_warp = tf.grid_sample(im, grid_normalized, align_corners=True)

    return x_warp


def energy_hs(im1, im2, flow, lambda_hs):
    """ Evalutes Horn-Schunck energy function.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow: torch tensor of shape (B, C, H, W)
        lambda_hs: float

    Returns:
        energy: torch tensor scalar
    """
    # reference: https://pytorch.org/docs/stable/generated/torch.autograd.grad.html

    # calculate the square of the difference(the quadratic penalties)
    im2_warp = warp_image(im2, flow)  # warp the second image by using the flow
    diff = torch.abs(im1 - im2_warp)
    # diff = im2 - im1
    QP = torch.sum(torch.pow(diff, 2))

    # calculate the pairwise MRF prior
    flow_grad_x = flow[:, 0, :, :]  # u
    flow_grad_y = flow[:, 1, :, :]  # v
    flow_grad_x.requires_grad_()
    flow_grad_y.requires_grad_()
    filter_x = torch.tensor([[[[0, 0, 0], [-1, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
    filter_y = torch.tensor([[[[0, -1, 0], [0, 1, 0], [0, 0, 0]]]], dtype=torch.float32)
    flow_gradient_x = torch.nn.functional.conv2d(flow_grad_x.unsqueeze(1), filter_x)
    flow_gradient_y = torch.nn.functional.conv2d(flow_grad_y.unsqueeze(1), filter_y)
    MRF = torch.sum(torch.pow(flow_gradient_x, 2) + torch.pow(flow_gradient_y, 2))

    # calculate the sum of energy
    energy = torch.sum(QP) + lambda_hs * torch.sum(MRF)

    return energy


def estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter):
    """
    Estimate flow using HS with Gradient Descent.
    Displays average endpoint error.
    Visualizes flow field.

    Args:
        im1: torch tensor of shape (B, C, H, W)
        im2: torch tensor of shape (B, C, H, W)
        flow_gt: torch tensor of shape (B, C, H, W)
        lambda_hs: float
        learning_rate: float
        num_iter: int

    Returns:
        aepe: torch tensor scalar
    """
    flow_estimate = torch.zeros_like(flow_gt)
    flow_estimate.requires_grad_(True)
    print('start AEPE evaluation before optimization......')
    aepe_before = evaluate_flow(flow_estimate, flow_gt)
    print('AEPE before optimization:', aepe_before)
    print('start flow estimation......')
    for i in range(num_iter):
        # calculate the energy using flow_estimate from last iteration
        energy = energy_hs(im1, im2, flow_estimate, lambda_hs)
        # print('energy:', energy)

        # calculate the gradients using autograd
        gradients = torch.autograd.grad(energy, flow_estimate, create_graph=True)

        # Update flow estimate using gradient descent
        with torch.no_grad():
            flow_estimate -= learning_rate * gradients[0]

    print('the minimum energy is:', energy)
    print('start AEPE evaluation after optimization......')
    aepe = evaluate_flow(flow_estimate, flow_gt)
    print('AEPE after optimization:', aepe)

    # convert the tensor back to ndarray
    flow_estimate.requires_grad_(False)
    flow_estimate = np.squeeze(flow_estimate, axis=0)
    flow_estimate_np = torch2numpy(flow_estimate)

    # visualization of estimated flow
    flow_rgb = flow2rgb(flow_estimate_np, max_value=None)
    fig, ax = plt.subplots()
    ax.imshow(flow_rgb)
    ax.set_title('Estimated flow after optimization')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.show()

    return aepe

# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():

    # Loading data
    im1, im2, flow_gt = load_data("data/frame10.png", "data/frame11.png", "data/flow10.flo")

    # Parameters
    lambda_hs = 0.002
    num_iter = 500

    # Warping_practice
    visualize_warping_practice(im1, im2, flow_gt)

    # Gradient descent
    learning_rate = 18
    estimate_flow(im1, im2, flow_gt, lambda_hs, learning_rate, num_iter)


if __name__ == "__main__":
    main()
    """
      --------------------test------------------------
    """
    # test numpy2torch() and torch2numpy()
    # numpy_array = np.random.rand(100, 200, 3)  # Create a random numpy array with HWC format
    # torch_tensor = numpy2torch(numpy_array)  # Convert numpy array to PyTorch tensor
    # numpy_array_converted = torch2numpy(torch_tensor)  # Convert PyTorch tensor back to numpy array
    #
    # print(numpy_array.shape)  # (100, 200, 3)
    # print(torch_tensor.shape)  # torch.Size([3, 100, 200])
    # print(numpy_array_converted.shape)  # (100, 200, 3)
    #
    # # test the load_data()
    # im1, im2, flow_gt = load_data("data/frame10.png", "data/frame11.png", "data/flow10.flo")
    # print(im1.shape)  # (100, 200, 3)
    # print(im2.shape)  # torch.Size([3, 100, 200])
    # print(flow_gt.shape)  # (100, 200, 3)
    # im1_squeeze = np.squeeze(im1, axis=0)
    # print(im1_squeeze.shape)
    #
    # # test warp_image()
    # visualize_warping_practice(im1, im2, flow_gt)
    #
    # # test energy_hs()
    # lambda_hs = 0.002
    # num_iter = 500
    # energy = energy_hs(im1, im2, flow_gt, lambda_hs)
    # print('energy:', energy)
    # print('evaluate_flow:',evaluate_flow(flow, flow_gt))