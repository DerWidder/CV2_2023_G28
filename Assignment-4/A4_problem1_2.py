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
from torchvision import transforms


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
        self.seg_train = os.path.join(root,
                                      'ImageSets/Segmentation/train.txt')  # segmentation results used for train set, names stored in a txt file
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

    # convert the color image into YCbCr color space
    ycbcr_image = skimage.color.rgb2ycbcr(np_image)
    H, W, C = np.shape(ycbcr_image)  # ycbcr_image:  numpy array (H,W,3) (float)
    # print(H,W,C)                  # 3 channels: [luma, Cb, Cr]

    # separate out channel 1: [luma]
    ycbcr_image_luma = ycbcr_image[:, :, 0]
    # initialization of channel 2/3 : [Cb, Cr]
    ycbcr_image_color = np.zeros_like(ycbcr_image[:, :, 1:3])
    # find the corresponding label by looking up the table VOC_LABEL2COLOR
    label_masks = [np_label == num for num, _ in enumerate(VOC_LABEL2COLOR)]
    # convert the label color from RGB to YCbCr
    color_values = np.array([skimage.color.rgb2ycbcr(np.array(val) / 255) for _, val in enumerate(VOC_LABEL2COLOR)])[:,
                   1:3]
    # sum the products of the color_values and label_masks
    ycbcr_image_color[:, :, 0] = np.sum(color_values[:, 0][:, None, None] * label_masks, axis=0)  # Cb
    ycbcr_image_color[:, :, 1] = np.sum(color_values[:, 1][:, None, None] * label_masks, axis=0)  # Cr

    # set the ambiguous color to (224, 224, 192)
    color_ambiguous = skimage.color.rgb2ycbcr(np.array([[[224, 224, 192]]]) / 255)
    # find the ambiguous label
    ycbcr_image_color[:, :, 0][np_label > 20] = color_ambiguous[0, 0, 1]
    ycbcr_image_color[:, :, 1][np_label > 20] = color_ambiguous[0, 0, 2]

    # recombined the texture from given image with the color from labels
    ycbcr_image_recombined = np.dstack((ycbcr_image_luma, ycbcr_image_color))

    # convert the color-space from YCbCr back to RGB
    colored = np.array(skimage.color.ycbcr2rgb(ycbcr_image_recombined), dtype=np.float32)

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
    # ref: https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.subplots.html
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.clip.html

    # initialization of figure, axes and title
    fig, axs = plt.subplots(grid_height, grid_width)
    fig.suptitle(title)

    # load the image from loader
    image_num = grid_width * grid_height  # the number of images to be shown
    for num, values in enumerate(loader):
        if num >= image_num:
            break
        # get the colored image using voc_label2color
        np_image = torch2numpy(values['im'][0]) / 255.0
        np_label = torch2numpy(values['gt'][0])
        np_colored_image = voc_label2color(np_image, np_label[:, :, 0])
        # Limit the value of the image to the range (0, 1) to avoid errors
        np_colored_image = np.clip(np_colored_image, 0, 1)
        # get row and colum from the quotient and remainder
        row, column = divmod(num, grid_width)
        ax = axs[row, column]
        # show the figure
        ax.imshow(np_colored_image)
        ax.axis('off')

    # show the figure
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
    # ref: https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html
    # ref: https://zhuanlan.zhihu.com/p/414242338

    mean = torch.tensor(VOC_STATISTICS['mean']).reshape(1, 3, 1, 1)
    std = torch.tensor(VOC_STATISTICS['std']).reshape(1, 3, 1, 1)
    # normalization using transform package from torchvision
    normalized = transforms.Normalize(mean, std)(input_tensor)

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
    # ref: https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
    # ref: https://www.geeksforgeeks.org/what-is-with-torch-no_grad-in-pytorch/

    # Put the model into evaluation mode
    model.eval()

    # disable the calculation of the gradient with torch.no_grad() to speed up the operation
    with torch.no_grad():
        # forward pass to get the model's output activations
        acts = model(normalized)['out']     # only consider the 'out', not the 'aux'

    # Obtain the model's confidence or probability estimate for each class
    probabilities = torch.softmax(acts, dim=1)
    # Get the final prediction labels
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
    # ref: https: // pytorch.org / docs / stable / generated / torch.unsqueeze.html

    # initialization of figure, axes and title
    fig, axs = plt.subplots(grid_height, grid_width)
    fig.suptitle(title)

    # load the image from loader
    image_num = grid_width * grid_height  # the number of images to be shown
    for num, values in enumerate(loader):
        if num >= image_num:
            break
        # get the colored image and label and convert it into numpy arrays
        np_image = torch2numpy(values['im'][0]) / 255.0
        np_label = torch2numpy(values['gt'][0])
        # Limit the value of the image to the range (0, 1) to avoid errors
        np_image = np.clip(np_image, 0, 1)
        # get the normalized image using the function we defined before
        normalized_image = normalize_input(torch.unsqueeze(numpy2torch(np_image), 0))  # unsqueeze is used for dimension expansion
        prediction, acts = run_forward_pass(normalized_image, model)
        # convert the prediction into numpy arrays
        np_prediction = torch2numpy(prediction[0])

        # get the colored image both from label and from prediction
        np_colored_image_label = voc_label2color(np_image, np_label[:, :, 0])
        np_colored_image_label = np.clip(np_colored_image_label, 0, 1)
        np_colored_image_prediction = voc_label2color(np_image, np_prediction[:, :, 0])
        np_colored_image_prediction = np.clip(np_colored_image_prediction, 0, 1)

        # combined the labels with the prediction results
        prediction_with_label = np.hstack((np_colored_image_label, np_colored_image_prediction))

        # computes the percentage of correctly labeled pixels
        avg_prec = average_precision(prediction, values['gt'])

        # get row and colum from the quotient and remainder
        row, column = divmod(num, grid_width)
        ax = axs[row, column]
        # show the figure
        ax.set_title("avg_prec = {}".format(avg_prec))  # put the performance metric into the title
        ax.imshow(prediction_with_label)
        ax.axis('off')

    # plt.tight_layout()
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
    # ref: https://stackoverflow.com/questions/50792316/what-does-1-mean-in-pytorch-view

    # flatten the prediction and ground truth tensors
    prediction_flat = prediction.view(-1)
    gt_flat = gt.view(-1)

    # compute the number of correctly labeled pixels
    num_correct = torch.sum(prediction_flat == gt_flat)
    # Calculate the total number of pixels
    num_correct_total = prediction_flat.size(0)
    # Compute the average precision
    avg_prec = num_correct.float() / num_correct_total

    return avg_prec


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
    # ref: https://numpy.org/doc/stable/reference/generated/numpy.unique.html

    example = []
    for num, values in enumerate(loader):
        np_label = torch2numpy(values['gt'][0])
        label_unique = np.unique(np_label[:,:,0])
        # Pick out examples that only contain label of 0 and the unique_foreground_label (also include the ambiguous label)
        if (len(label_unique) == 3) and (0 in label_unique) and (255 in label_unique) and (unique_foreground_label in label_unique):
            example = values
            break

    assert (isinstance(example, dict))
    return example


def show_unique_example(example_dict, model):
    """Visualise the results produced for a given sample (see Fig. 3).

    Args:
        example_dict: a dict with keys 'gt' and 'im' returned by an instance of VOC2007Dataset
        model: network (nn.Module)
    """
    # similar to problem 1

    # initialization of figure, axes and title
    fig, axs = plt.subplots(1, 1)
    fig.suptitle('unique examples (before fooling)')

    np_image = torch2numpy(example_dict['im'][0]) / 255.0
    np_label = torch2numpy(example_dict['gt'][0])
    normalized_image = normalize_input(torch.unsqueeze(numpy2torch(np_image), 0))  # unsqueeze is used for dimension expansion
    prediction, acts = run_forward_pass(normalized_image, model)
    # convert the prediction into numpy arrays
    np_prediction = torch2numpy(prediction[0])

    # get the colored image both from label and from prediction
    np_colored_image_label = voc_label2color(np_image, np_label[:, :, 0])
    np_colored_image_label = np.clip(np_colored_image_label, 0, 1)
    np_colored_image_prediction = voc_label2color(np_image, np_prediction[:, :, 0])
    np_colored_image_prediction = np.clip(np_colored_image_prediction, 0, 1)

    # combined the labels with the prediction results
    prediction_with_label = np.hstack((np_colored_image_label, np_colored_image_prediction))

    # computes the percentage of correctly labeled pixels
    avg_prec = average_precision(prediction, example_dict['gt'])

    # show the figure
    axs.set_title("avg_prec = {}".format(avg_prec))  # put the performance metric into the title
    axs.imshow(prediction_with_label)
    axs.axis('off')
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
    pass


# Example usage in main()
# Feel free to experiment with your code in this function
# but make sure your final submission can execute this code
def main():
    # Please set an environment variables 'VOC2007_HOME' pointing to your '../VOCdevkit/VOC2007' folder

    root = 'VOCtrainval_06-Nov-2007/VOCdevkit/VOC2007/'
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
    show_inference_examples(valid_loader, model, grid_height=2, grid_width=3, title='inference examples')

    # attack1: convert cat to dog
    cat_example = find_unique_example(valid_loader, unique_foreground_label=8)
    show_unique_example(cat_example, model=model)
    show_attack(cat_example, model, src_label=8, target_label=12, learning_rate=1.0, iterations=10)

    # feel free to try other examples..


if __name__ == '__main__':
    main()
