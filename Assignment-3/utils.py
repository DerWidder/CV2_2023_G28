from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

import numpy as np
from PIL import Image
#from scipy import ndimage

UNKNOWN_FLOW_THRESH = 1e9


def rgb2gray(rgb):
    """Converts Numpy 3D color images to grayscale."""
    return np.expand_dims(np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140]), axis=2).astype(rgb.dtype)


def read_image(filename):
    """Reads a color image from filename and returns it as numpy ndarray."""

    return (np.array(Image.open(filename))/255).astype(np.float32)


def read_flo(filename):
    """Reads optical flow from .flo file and returns it as numpy ndarray."""
    with open(filename, 'rb') as file:
        magic = np.fromfile(file, np.float32, count=1)
        assert (
            202021.25 == magic), "Magic number incorrect. Invalid .flo file"
        w = np.fromfile(file, np.int32, count=1)[0]
        h = np.fromfile(file, np.int32, count=1)[0]
        data = np.fromfile(file, np.float32, count=2 * h * w)
    data_2d = np.resize(data, (h, w, 2))
    return data_2d


def flow2rgb(flow, max_value=None):
    """Converts a given optical flow field to RGB colors.

      Args:
        flow: A `numpy.ndarray`. Typically of the following types: `float32`, `float64`.
          Dimension is expected to be MxNx2, M/N is the number of rows/cols, respectively, and the third dimension
          contains the horizontal/vertical components of the flow field.

        max_value: An (optional) maximum absolute flow to be used to normalize the flow visualization.
          If max_value is not specified, maximum absolute flow is computed from the given flow field.

      Returns:
        A `nd.array` of type `uint8`. Color representation of the given optical flow field.
      """

    assert (isinstance(flow, np.ndarray) and flow.dtype in [np.float32, np.float64])
    assert (flow.shape[2] == 2)

    height, width, dim = flow.shape
    if dim != 2:
        print('ERROR: flowToColor: image must have two bands')
        return np.zeros((height, width, 3))
    u = flow[:, :, 0]
    v = flow[:, :, 1]
    # fix unknown flow
    idx_unknown = np.logical_or(
        np.isnan(u),
        np.logical_or(
            np.isnan(v),
            np.logical_or(
                abs(u) > UNKNOWN_FLOW_THRESH,
                abs(v) > UNKNOWN_FLOW_THRESH)))
    u[idx_unknown] = 0
    v[idx_unknown] = 0
    if max_value is not None:
        maxrad = max_value
    else:
        rad = np.sqrt(u ** 2 + v ** 2)
        maxrad = max(-1, rad.max())
    if maxrad > 0:
        u = u / maxrad
        v = v / maxrad
    # compute color
    img = _compute_color(u, v)
    return img


def _compute_color(u, v):
    """Helper function."""
    nan_idx = np.zeros((u.shape[0], u.shape[1]))
    colorwheel = _make_colorwheel()
    ncols = colorwheel.shape[0]
    rad = np.sqrt(u ** 2 + v ** 2)
    a = np.arctan2(-v, -u) / math.pi
    fk = np.maximum((a + 1) / 2 * (ncols - 1), 0)  # -1~1 maped to 1~ncols-1
    k0 = np.int_(np.floor(fk))  # 0, 1, ..., ncols-1
    k1 = np.mod(k0 + 1, ncols)
    f = fk - k0
    img = np.zeros((u.shape[0], u.shape[1], 3)).astype('uint8')
    for i in range(0, colorwheel.shape[1]):
        tmp = colorwheel[:, i]
        col0 = tmp[k0] / 255
        col1 = tmp[k1] / 255
        col = (1 - f) * col0 + f * col1
        idx = rad <= 1
        col[idx] = 1 - rad[idx] * (1 - col[idx])  # increase saturation with radius
        nidx = rad > 1
        col[nidx] = col[nidx] * 0.75  # out of range
        img[:, :, i] = np.uint8(np.floor(255 * col * (1 - nan_idx)))
    return img


def _make_colorwheel():
    """Helper function."""
    ry = 15
    yg = 6
    gc = 4
    cb = 11
    bm = 13
    mr = 6
    ncols = ry + yg + gc + cb + bm + mr
    colorwheel = np.zeros((ncols, 3))  # r g b
    col = 0
    # ry
    colorwheel[0:ry, 0] = 255
    colorwheel[0:ry, 1] = np.transpose(np.floor(255 * (np.arange(0, ry)) / ry))
    col = col + ry
    # YG
    colorwheel[col + np.arange(0, yg), 0] = np.transpose(
        255 - np.floor(255 * (np.arange(0, yg)) / yg))
    colorwheel[col + np.arange(0, yg), 1] = 255
    col = col + yg
    # GC
    colorwheel[col + np.arange(0, gc), 1] = 255
    colorwheel[col + np.arange(0, gc), 2] = np.transpose(
        np.floor(255 * (np.arange(0, gc)) / gc))
    col = col + gc
    # CB
    colorwheel[col + np.arange(0, cb), 1] = np.transpose(
        255 - np.floor(255 * (np.arange(0, cb)) / cb))
    colorwheel[col + np.arange(0, cb), 2] = 255
    col = col + cb
    # BM
    colorwheel[col + np.arange(0, bm), 2] = 255
    colorwheel[col + np.arange(0, bm), 0] = np.transpose(
        np.floor(255 * (np.arange(0, bm)) / bm))
    col = col + bm
    # MR
    colorwheel[col + np.arange(0, mr), 2] = np.transpose(
        255 - np.floor(255 * (np.arange(0, mr)) / mr))
    colorwheel[col + np.arange(0, mr), 0] = 255
    return colorwheel
