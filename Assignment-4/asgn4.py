from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import torch
from PIL import Image
import skimage

import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision.io import read_image

from matplotlib import pyplot as plt

from utils import VOC_LABEL2COLOR
from utils import VOC_STATISTICS
# Added import
import random
from utils import torch2numpy
from utils import numpy2torch

from torchvision import models

class VOC2007Dataset(Dataset):
    """
    Class to create a dataset for VOC 2007
    Refer to https://pytorch.org/tutorials/beginner/basics/data_tutorial.html for an instruction on PyTorch datasets.
    """

    def __init__(self, root, train, num_examples):
        super().__init__()
        """
        Initialize the dataset by setting the required attributes.

        Args:
            root: root folder of the dataset (string)
            train: indicates if we want to load training (train=True) or validation data (train=False)
            num_examples: size of the dataset (int)

        Returns (just set the required attributes):
            input_filenames: list of paths to individual images
            target_filenames: list of paths to individual segmentations (corresponding to input_filenames)
            rgb2label: lookup table that maps RGB values to class labels using the constants in VOC_LABEL2COLOR.
        """
        self.num_examples = num_examples
        self.img_root = os.path.join(root, 'JPEGImages/')
        self.seg_root = os.path.join(root, 'SegmentationClass/')  # folder with contains segmentation results
        self.seg_train = os.path.join(root, 'ImageSets/Segmentation/train.txt')  # segmentation results used for train set, names stored in a txt file
        self.seg_val = os.path.join(root, 'ImageSets/Segmentation/val.txt')
        img_name = []
        # name of images used for train or validation
        if train == True:
            with open(file=self.seg_train, mode='r', encoding='utf-8') as f:
                # append image names in the list and delete '\n' at end
                [img_name.append(i.strip()) for i in f.readlines()]
        else:
            with open(file=self.seg_val, mode='r', encoding='utf-8') as f:
                [img_name.append(i.strip()) for i in f.readlines()]

        # print('image names: ', img_name)
        # input_filenames: list of paths to individual images
        # self.img = [os.path.join(self.img_root, i) for i in img_name[:self.num_examples]]
        random_index = random.sample(img_name, num_examples)
        self.input_filenames = [os.path.join(self.img_root, i) for i in random_index]
        # print('input names: ', self.input_filenames)
        # target_filenames: list of paths to individual segmentations (corresponding to input_filenames)
        # self.seg = [os.path.join(self.seg_root, i) for i in img_name[:self.num_examples]]
        self.target_filenames = [os.path.join(self.seg_root, i) for i in random_index]

        # lookup table that maps RGB values to class labels using the constants in VOC_LABEL2COLOR.
        self.rgb2label = {color: i for i, color in enumerate(VOC_LABEL2COLOR)}

        # label2rgb = dict()
        # for i in range(256):
        #     if i < len(VOC_LABEL2COLOR):
        #         label2rgb[i] = VOC_LABEL2COLOR[i]
        #     else:
        #         label2rgb[i] = (224, 224, 192)
        # label2rgb = {idx: VOC_LABEL2COLOR[idx] if idx < len(VOC_LABEL2COLOR) else (224, 224, 192) for idx in range(len(VOC_LABEL2COLOR) + 235)}




    def __getitem__(self, index):
        """
        Return an item from the dataset.

        Args:
            index: index of the item (Int)

        Returns:
            item: dictionary of the form {'im': the_image, 'gt': the_label}
            with the_image being a torch tensor (3, H, W) (float) and 
            the_label being a torch tensor (1, H, W) (long) and 
        """
        item = dict()
        # the path to the image
        img_path = self.input_filenames[index] + '.jpg'
        # the path to the ground truth image segmentation
        gt_path = self.target_filenames[index] + '.png'

        img_torch = numpy2torch(np.array(Image.open(img_path))).to(torch.float32)
        gt_torch = numpy2torch(np.array(Image.open(gt_path))).to(torch.int64)

        # C, H, W = gt_torch.shape
        # gt_torch_covert = torch.full((1, H, W), 0)  # create a (1, H, W) tensor of gt_torch
        # # maps RGB values (3 channels) to class labels (1 channel) using the constants in VOC_LABEL2COLOR
        # for num, val in enumerate(VOC_LABEL2COLOR):
        #     # check the RGB values and link them to the label in VOC_LABEL2COLOR
        #     # checks if all elements along dimension 0 are True, returns a new tensor of shape (H, W)
        #     # mask = (gt_torch == torch.Tensor(val).view(3, 1, 1)).all(dim=0)
        #     # mask = torch.unsqueeze(mask, dim=0)
        #     # gt_torch_covert[mask] = num
        #     gt_torch_covert[gt_torch == torch.Tensor(val).view(3, 1, 1)] = num

        item['im'] = img_torch
        item['gt'] = gt_torch

        assert (isinstance(item, dict))
        assert ('im' in item.keys())
        assert ('gt' in item.keys())

        return item

    def __len__(self):
        """
        Return the length of the dataset.

        Args:

        Returns:
            length: length of the dataset (int)
        """
        return len(self.input_filenames)



def create_loader(dataset, batch_size, shuffle, num_workers=1):
    """
    Return loader object.

    Args:
        dataset: PyTorch Dataset
        batch_size: int
        shuffle: bool
        num_workers: int

    Returns:
        loader: PyTorch DataLoader
    """
    # ref: https://pytorch.org/tutorials/beginner/basics/data_tutorial.html
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers)
    assert (isinstance(loader, DataLoader))
    return loader


def voc_label2color(np_image, np_label):
    """
    Super-impose labels on a given image using the colors defined in VOC_LABEL2COLOR.

    Args:
        np_image: numpy array (H,W,3) (float)
        np_label: numpy array  (H,W) (int)  # each pixel has a label:0~20, the corresponding color(r,g,b) is stored in VOC_LABEL2COLOR

    Returns:
        colored: numpy array (H,W,3) (float)
    """
    # ref: https://scikit-image.org/docs/stable/api/skimage.color.html#skimage.color.label2rgb
    assert (isinstance(np_image, np.ndarray))
    assert (isinstance(np_label, np.ndarray))
    '''
    另外一组的代码 要做修改
    '''
    # Convert color image to YCbCr color space
    img_ycbcr = skimage.color.rgb2ycbcr(np_image)  # same shape(H, W, 3)

    '''
        y是亮度分量，就是texture? 
        cb蓝色分量，cr红色分量
    '''
    # img_texture = img_ycbcr[:, :, 0]
    # color_channels = np.zeros_like(img_ycbcr[:, :, 1:])  # Initialize color channels  # shape(H, W, 2)

    for num, val in enumerate(VOC_LABEL2COLOR):
        # Find pixels which have the same label
        # label_pixels = np_label == num
        # print('ndim of the val array: ', np.array(val).ndim)
        # color = skimage.color.rgb2ycbcr(np.array(val) / 255)  # RGB value should be inside (0, 1) and shape of RGB should be 3D
        color = skimage.color.rgb2ycbcr(np.array([[[val[0], val[1], val[2]]]]) / 255)  # RGB value should be inside (0, 1) and shape of RGB should be 3D
        # print('shape of ycbcr color ', color.shape)  # shape of color (1, 1, 3)
        # y, cb, cr = color
        # color_channels[:, :, 0][label_pixels] = cb
        # color_channels[:, :, 1][label_pixels] = cr
        img_ycbcr[:, :, 1][np_label == num] = color[0, 0, 1]  # change the second channel for cb color
        img_ycbcr[:, :, 2][np_label == num] = color[0, 0, 2]  # change the third channel for cr color


    # indicates an ambiguous label (i.e. neither an object nor the background), the num of label is larger than 20
    # label_pixels = np_label == 255
    color_ambiguous = skimage.color.rgb2ycbcr(np.array([[[224, 224, 192]]]) / 255)
    # y_ambiguous, cb_ambiguous, cr_ambiguous = color_ambiguous
    img_ycbcr[:, :, 1][np_label > 20] = color_ambiguous[0, 0, 1]
    img_ycbcr[:, :, 2][np_label > 20] = color_ambiguous[0, 0, 2]

    # ycbcr_modified = np.dstack((img_texture, color_channels))

    # Convert color-coded representation back to RGB
    # colored = skimage.color.ycbcr2rgb(ycbcr_modified)
    # print('dtype of np_image: ', np_image.dtype)
    colored = np.array(skimage.color.ycbcr2rgb(img_ycbcr), dtype=np.float32)

    assert (np.equal(colored.shape, np_image.shape).all())
    assert (np_image.dtype == colored.dtype)
    return colored


def show_dataset_examples(loader, grid_height, grid_width, title):
    """
    Visualize samples from the dataset.

    Args:
        loader: PyTorch DataLoader
        grid_height: int
        grid_width: int
        title: string
    """
    '''
        另外一组的代码 要做修改
    '''
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(18, 18))
    fig.suptitle(title, fontsize=30)

    for num, val in enumerate(loader):
        # only use number of samples limited by grid size
        if num < grid_width*grid_height:
            # shape of im: (B, C, H, W), we only need (C, H, W)
            # shape of label: (B, C, H, W), we only need (C, H, W)
            image = val['im'][0]
            label = val['gt'][0]
            # print('val in image: ', image[1, 1, 5])  # check the value of pixel, is between [0, 255]
            # print('val in label: ', label[0, 1, 5])  # check the value of label
            # print('image shape: ', image.shape)
            # print('label shape', label.shape)

            # Convert tensors to NumPy arrays
            image_np = torch2numpy(image) / 255.0  # scale the pixel value in [0, 1]
            # print(image_np.shape)

            label_np = torch2numpy(label)
            # print('label_np shape', label_np.shape)

            # Apply voc_label2color to get color-coded labels
            colored_label = voc_label2color(image_np, label_np[:, :, 0])
            # colored_label = voc_label2color(image_np, label_np)
            colored_label = np.clip(colored_label, 0, 1)
            # Plot the image and colored label
            row = num // grid_width
            col = num % grid_width
            axs[row, col].imshow(colored_label)
            axs[row, col].axis('off')
        else:
            break


    plt.tight_layout()
    plt.show()
    pass

def normalize_input(input_tensor):
    """
    Normalize a tensor using statistics in VOC_STATISTICS.

    Args:
        input_tensor: torch tensor (B,3,H,W) (float32)
        
    Returns:
        normalized: torch tensor (B,3,H,W) (float32)
    """
    # ref: ChatGPT
    mean = torch.tensor(VOC_STATISTICS['mean']).reshape(1, 3, 1, 1)
    std = torch.tensor(VOC_STATISTICS['std']).reshape(1, 3, 1, 1)
    normalized = (input_tensor - mean) / std

    assert (type(input_tensor) == type(normalized))
    assert (input_tensor.size() == normalized.size())
    return normalized

def run_forward_pass(normalized, model):
    """
    Run forward pass.

    Args:
        normalized: torch tensor (B,3,H,W) (float32)
        model: PyTorch model
        
    Returns:
        prediction: class prediction of the model (B,1,H,W) (int64)
        acts: activations of the model (B,21,H,W) (float 32)
    """
    # ref: ChatGPT
    # Put the model into evaluation mode
    model.eval()

    # Forward pass to get the model's output activations
    with torch.no_grad():
        # acts = model(normalized)
        # forward pass to get the model's output activations
        acts = model(normalized)['out']  # only consider the 'out', not the 'aux'

    # Obtain the model's confidence or probability estimate for each class
    probabilities = torch.softmax(acts, dim=1)
    # Get the final prediction labels
    # _, prediction = torch.max(acts, dim=1)
    _, prediction = torch.max(probabilities, dim=1)
    prediction = torch.unsqueeze(prediction, 0)

    assert (isinstance(prediction, torch.Tensor))
    assert (isinstance(acts, torch.Tensor))
    return prediction, acts

def show_inference_examples(loader, model, grid_height, grid_width, title):
    """
    Perform inference and visualize results.

    Args:
        loader: PyTorch DataLoader
        model: PyTorch model
        grid_height: int
        grid_width: int
        title: string
    """
    '''
        另外一组的代码 要做修改
        '''
    fig, axs = plt.subplots(grid_height, grid_width, figsize=(18, 18))
    fig.suptitle(title)

    for num, val in enumerate(loader):
        if num >= grid_height * grid_width:
            break

        image = val['im'][0]
        gt = val['gt'][0]

        # Convert tensors to NumPy arrays
        np_image = torch2numpy(image) / 255.0
        np_label = torch2numpy(gt)
        np_image = np.clip(np_image, 0, 1)
        normalized_image = normalize_input(torch.unsqueeze(numpy2torch(np_image), 0)) # unsqueeze is used for dimension expansion
        # label_np = torch2numpy(gt)
        prediction, acts = run_forward_pass(normalized_image, model)

        np_prediction = torch2numpy(prediction[0])
        # get the colored image both from label and from prediction
        np_colored_image_label = voc_label2color(np_image, np_label[:, :, 0])
        np_colored_image_label = np.clip(np_colored_image_label, 0, 1)
        np_colored_image_prediction = voc_label2color(np_image, np_prediction[:, :, 0])
        np_colored_image_prediction = np.clip(np_colored_image_prediction, 0, 1)

        # combined the labels with the prediction results
        prediction_with_label = np.hstack((np_colored_image_label, np_colored_image_prediction))
        # prediction_with_label = np.hstack((np_label, np_prediction))[:, :, 0]
        # print('shape of prediction_with_label: ', prediction_with_label.shape)
        # computes the percentage of correctly labeled pixels
        avg_prec = average_precision(prediction, val['gt'])

        row = num // grid_width
        col = num % grid_width
        # axs[row, col].set_title(f"avg_prec ={avg_prec} ")
        axs[row, col].set_title("avg_prec = {}".format(avg_prec))  # put the performance metric into the title
        axs[row, col].imshow(prediction_with_label)
        axs[row, col].axis('off')
        # get row and colum from the quotient and remainder
        # row, column = divmod(num, grid_width)  # divmod 返回一个包含商和余数的元组
        # ax = axs[row, column]
        # # show the figure
        # ax.set_title("avg_prec = {}".format(avg_prec))  # put the performance metric into the title
        # ax.imshow(prediction_with_label)
        # ax.axis('off')

    plt.tight_layout()
    plt.show()
    pass

def average_precision(prediction, gt):
    """
    Compute percentage of correctly labeled pixels.

    Args:
        prediction: torch tensor (B,1,H,W) (int)
        gt: torch tensor (B,1,H,W) (int)
       
    Returns:
        avg_prec: torch scalar (float32)
    """
    # Flatten the prediction and ground truth tensors
    # prediction_flat = prediction.flatten()
    # gt_flat = gt.flatten()
    prediction_flat = prediction.view(-1)
    gt_flat = gt.view(-1)

    # compute the number of correctly labeled pixels
    num_correct = torch.sum(prediction_flat == gt_flat)
    # Calculate the total number of pixels
    num_correct_total = prediction_flat.size(0)
    # Compute the average precision
    avg_prec = num_correct.float() / num_correct_total

    return avg_prec
    # # Calculate the number of correctly labeled pixels
    # # correct_pixels = np.sum(prediction_flat == gt_flat)
    #
    # # Calculate the total number of pixels
    # total_pixels = prediction_flat.size

    # Compute the average precision
    # avg_precision = correct_pixels / total_pixels

    # return avg_precision


### FUNCTIONS FOR PROBLEM 2 ###

def find_unique_example(loader, unique_foreground_label):
    """Returns the first sample containing (only) the given label

    Args:
        loader: dataloader (iterable)
        unique_foreground_label: the label to search

    Returns:
        sample: a dictionary with keys 'im' and 'gt' specifying
                the image sample 
    """
    # set the background value
    background_label = 0
    example = dict()
    for num, val in enumerate(loader):
        np_label = torch2numpy(val['gt'][0])
        # get the foreground labels
        foreground_label = set(np_label.flatten()) - {background_label}

        if len(foreground_label) == 1 and unique_foreground_label in foreground_label:
            example['im'] = val['im']
            example['gt'] = val['gt']
            # example = val
            break

    assert (isinstance(example, dict))
    return example


def show_unique_example(example_dict, model):
    """Visualise the results produced for a given sample (see Fig. 3).

    Args:
        example_dict: a dict with keys 'gt' and 'im' returned by an instance of VOC2007Dataset
        model: network (nn.Module)
    """
    fig, axs = plt.subplots(1, 1)

    image = example_dict['im'][0]
    gt = example_dict['gt'][0]

    # Convert tensors to NumPy arrays
    np_image = torch2numpy(image) / 255.0
    np_label = torch2numpy(gt)
    np_image = np.clip(np_image, 0, 1)
    normalized_image = normalize_input(torch.unsqueeze(numpy2torch(np_image), 0))  # unsqueeze is used for dimension expansion
    # label_np = torch2numpy(gt)
    prediction, acts = run_forward_pass(normalized_image, model)

    np_prediction = torch2numpy(prediction[0])
    # get the colored image both from label and from prediction
    np_colored_image_label = voc_label2color(np_image, np_label[:, :, 0])
    np_colored_image_label = np.clip(np_colored_image_label, 0, 1)
    np_colored_image_prediction = voc_label2color(np_image, np_prediction[:, :, 0])
    np_colored_image_prediction = np.clip(np_colored_image_prediction, 0, 1)

    # combined the labels with the prediction results
    prediction_with_label = np.hstack((np_colored_image_label, np_colored_image_prediction))
    # prediction_with_label = np.hstack((np_label, np_prediction))[:, :, 0]
    # print('shape of prediction_with_label: ', prediction_with_label.shape)
    # computes the percentage of correctly labeled pixels
    avg_prec = average_precision(prediction, example_dict['gt'])


    # axs[row, col].set_title(f"avg_prec ={avg_prec} ")
    axs[1, 1].set_title("avg_prec = {}".format(avg_prec))  # put the performance metric into the title
    axs[1, 1].imshow(prediction_with_label)
    axs[1, 1].axis('off')


    plt.tight_layout()
    plt.show()
    pass


def show_attack(example_dict, model, src_label, target_label, learning_rate, iterations):
    """Modify the input image such that the model prediction for label src_label
    changes to target_label.

    Args:
        example_dict: a dict with keys 'gt' and 'im' returned by an instance of VOC2007Dataset
        model: network (nn.Module)
        src_label: the label to change
        target_label: the label to change to
        learning_rate: the learning rate of optimisation steps
        iterations: number of optimisation steps

    This function does not return anything, but instead visualises the results (see Fig. 4).
    """
    tensor_image = example_dict['im']
    tensor_label = example_dict['gt']
    np_image = torch2numpy(example_dict['im'][0]) / 255.0
    np_label = torch2numpy(example_dict['gt'][0])

    # convert all pixels with a src label to a target label
    # fake_np_label = np.copy(np_label)
    # fake_np_label[fake_np_label == src_label] = target_label
    fake_tensor_label = tensor_label.copy()
    fake_tensor_label[fake_tensor_label == src_label] == target_label

    # enable the gradient tracking for an input image
    tensor_image.requires_grad = True

    # run the standard pipeline consisting of standardization and model forward pass
    # normalized_tensor_image = normalize_input(tensor_image)
    # prediction, acts = run_forward_pass(normalized_tensor_image, model)

    # apply a cross entropy to compute the loss of the predictions w.r.t. the fake_tensor_label
    # loss_cross_entropy = torch.nn.CrossEntropyLoss(prediction, fake_tensor_label)

    # set pixels corresponding to background (in the original ground truth) to zero
    changed_tensor_image = tensor_image.copy()
    changed_tensor_image[tensor_label == 0] = 0

    # define the optimizer
    optimizer = optim.LBFGS(changed_tensor_image, lr=learning_rate)
    def closure():
        # reset the gradients
        optimizer.zero_grad()

        # forward pass
        normalized_tensor_image = normalize_input(tensor_image)
        prediction, acts = run_forward_pass(normalized_tensor_image, model)

        # compute cross entropy
        loss_cross_entropy = torch.nn.CrossEntropyLoss(prediction, fake_tensor_label)
        loss_cross_entropy.backward()

    # optimization loop
    for _ in range(iterations):
        optimizer.step(closure)

    # Updated input image
    updated_image = tensor_image.detach()

    # visualize tensor_image, updated_image, difference and prediction

    pass


# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():
    # Please set an environment variables 'VOC2007_HOME' pointing to your '../VOCdevkit/VOC2007' folder

    root = 'D:/Daten/TUD SS2020/Vorlesung/CV2/Assignments/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/'
    # os.environ["VOC2007_HOME"] = 'D:/Daten/TUD SS2020/Vorlesung/CV2/Assignments/VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007'
    # root = os.environ["VOC2007_HOME"]

    # create datasets for training and validation
    train_dataset = VOC2007Dataset(root, train=True, num_examples=128)
    valid_dataset = VOC2007Dataset(root, train=False, num_examples=128)

    # create data loaders for training and validation
    train_loader = create_loader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    valid_loader = create_loader(valid_dataset, batch_size=1, shuffle=False, num_workers=1)

    # show some images for the training and validation set
    show_dataset_examples(train_loader, grid_height=2, grid_width=3, title='training examples')
    show_dataset_examples(valid_loader, grid_height=2, grid_width=3, title='validation examples')

    # Load FCN network
    model = models.segmentation.fcn_resnet101(pretrained=True, num_classes=21)

    # Apply fcn. Switch to training loader if you want more variety.
    # show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')

    # attack1: convert cat to dog
    cat_example = find_unique_example(valid_loader, unique_foreground_label=8)
    show_unique_example(cat_example, model=model)
    show_attack(cat_example, model, src_label=8, target_label=12, learning_rate=1.0, iterations=10)

    # feel free to try other examples..

if __name__ == '__main__':
    main()
