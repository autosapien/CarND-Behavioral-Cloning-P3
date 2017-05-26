import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import cv2
import time
import numpy as np
import scipy.ndimage as ndi
from math import ceil


def load_image(filename):
    return cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)


def show_image_color(img):
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def show_image_numpy_gray(img):
    img = ((img+0.5).squeeze() * 255).astype(np.uint8)
    cv2.namedWindow('image', cv2.WINDOW_NORMAL)
    cv2.imshow('image', cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def save_image(img, filename):
    cv2.imwrite(filename, cv2.cvtColor(img, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])


def image_rgb_equalize(img):
    img_yuv = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
    # equalize the histogram of the Y channel
    img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
    # convert the YUV image back to RGB format
    return cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)


def image_gray_equalize(img):
    # equalize the histogram of the input image
    histeq = cv2.equalizeHist(img)
    return histeq


def image_shift_horiz(x, shift_by, row_axis=0, col_axis=1, channel_axis=2,
                      fill_mode='nearest', cval=0.):
    ty = shift_by * x.shape[col_axis]
    shift_matrix = np.array([[1, 0, 0],
                             [0, 1, ty],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shift_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_shift_horiz(x, shift_by, row_axis=0, col_axis=1, channel_axis=2,
                      fill_mode='nearest', cval=0.):
    ty =  np.random.uniform(-shift_by, shift_by) * x.shape[col_axis]
    shift_matrix = np.array([[1, 0, 0],
                             [0, 1, ty],
                             [0, 0, 1]])
    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shift_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


def random_brightness(x, multiplier=(0.5, 1.5)):
    x = cv2.cvtColor(x, cv2.COLOR_RGB2HSV)
    factor = np.random.uniform(multiplier[0], multiplier[1])
    safety = ceil(255 / factor)
    x[:, :, 2] = np.where(x[:, :, 2] > safety, 255, (x[:, :, 2]*factor).astype(np.uint8))  # Took too much time :D
    return cv2.cvtColor(x, cv2.COLOR_HSV2RGB)


def image_shear(x, intensity, row_axis=0, col_axis=1, channel_axis=2,
                fill_mode='nearest', cval=0.):
    """Performs a random spatial shear of a Numpy image tensor.
       # Arguments
           x: Input tensor. Must be 3D.
           intensity: Transformation intensity.
           row_axis: Index of axis for rows in the input tensor.
           col_axis: Index of axis for columns in the input tensor.
           channel_axis: Index of axis for channels in the input tensor.
           fill_mode: Points outside the boundaries of the input
               are filled according to the given mode
               (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
           cval: Value used for points outside the boundaries
               of the input if `mode='constant'`.
       # Returns
           Sheared Numpy image tensor.
       """
    shear_matrix = np.array([[1, -np.sin(intensity), 0],
                             [0, np.cos(intensity), 0],
                             [0, 0, 1]])

    h, w = x.shape[row_axis], x.shape[col_axis]
    transform_matrix = transform_matrix_offset_center(shear_matrix, h, w)
    x = apply_transform(x, transform_matrix, channel_axis, fill_mode, cval)
    return x


# Taken from Keras https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
def transform_matrix_offset_center(matrix, x, y):
    o_x = float(x) / 2 + 0.5
    o_y = float(y) / 2 + 0.5
    offset_matrix = np.array([[1, 0, o_x], [0, 1, o_y], [0, 0, 1]])
    reset_matrix = np.array([[1, 0, -o_x], [0, 1, -o_y], [0, 0, 1]])
    transform_matrix = np.dot(np.dot(offset_matrix, matrix), reset_matrix)
    return transform_matrix


# Taken from Keras  https://github.com/fchollet/keras/blob/master/keras/preprocessing/image.py
def apply_transform(x,
                    transform_matrix,
                    channel_axis=0,
                    fill_mode='nearest',
                    cval=0.):
    """Apply the image transformation specified by a matrix.
    # Arguments
        x: 2D numpy array, single image.
        transform_matrix: Numpy array specifying the geometric transformation.
        channel_axis: Index of axis for channels in the input tensor.
        fill_mode: Points outside the boundaries of the input
            are filled according to the given mode
            (one of `{'constant', 'nearest', 'reflect', 'wrap'}`).
        cval: Value used for points outside the boundaries
            of the input if `mode='constant'`.
    # Returns
        The transformed version of the input.
    """
    x = np.rollaxis(x, channel_axis, 0)
    final_affine_matrix = transform_matrix[:2, :2]
    final_offset = transform_matrix[:2, 2]
    channel_images = [ndi.interpolation.affine_transform(
        x_channel,
        final_affine_matrix,
        final_offset,
        order=0,
        mode=fill_mode,
        cval=cval) for x_channel in x]
    x = np.stack(channel_images, axis=0)
    x = np.rollaxis(x, 0, channel_axis + 1)
    return x