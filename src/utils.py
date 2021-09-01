import numpy as np

from scipy import signal
from scipy.ndimage.filters import convolve


def getBaseLayer(img, kernel_size, sigma):
    """
    Get  base layer given an image
    :param img: input image
    :param kernel_size: kernel size
    :param sigma: standard deviation
    :return base: base layer of input image
    """
    h, w, c = img.shape

    color = np.zeros((h, w, c))
    intensity = 0.2989 * img[:, :, 0] \
        + 0.5870 * img[:, :, 1] \
        + 0.1140 * img[:, :, 2]

    for idx in range(c):
        color[:, :, idx] = img[:, :, idx] / (intensity + 1e-4)

    luminous = np.log2(intensity + 1e-4)
    padding = int(np.floor(kernel_size / 2))
    y, x = np.mgrid[- padding: padding + 1, - padding: padding + 1]
    kernel = np.exp(- (x ** 2 + y ** 2) / (2 * sigma ** 2))
    kernel = kernel / np.sum(kernel)  # normalize

    convolved = signal.convolve2d(luminous, kernel, boundary='symm',
                                  mode='same')
    new_intensity = np.power(2, convolved)

    base = np.zeros((h, w, c))
    for idx in range(c):
        base[:, :, idx] = color[:, :, idx] * (new_intensity - 1e-4)
    base = np.round(np.clip(base, 0, 255)).astype('uint8')

    return base


def getAvgMask(img, kernel_h, kernel_w):
    """
    Get smooth mask by averaging
    :param img: input image
    :param kernel_h: height of kernel
    :param kernel_w: weight of kernel
    :return averaged_mask: averaged mask of input image
    """
    h, w, c = img.shape

    intensity = np.average(img, 2)
    luminous = np.log2(intensity + 1e-4)
    padding_h = int(np.floor(kernel_h / 2))
    padding_w = int(np.floor(kernel_w / 2))
    luminous = np.pad(luminous,
                      ((padding_h, padding_h), (padding_w, padding_w)),
                      'symmetric')

    kernel = np.ones((kernel_h, kernel_w))

    averaged_mask = np.zeros((h, w))
    for i in range(h):
        for j in range(w):
            averaged_mask[i, j] = np.sum(
                luminous[i: i + kernel_h, j: j + kernel_w] * kernel) / kernel.size

    return averaged_mask


def getEnergyMap(img, mask):
    """
    Get energy map of input image according to a mask
    :param img: input image
    :param mask: input mask
    :return energy_map: energy map of input image
    """
    filter_u = np.array([[1.0, 2.0, 1.0], [0.0, 0.0, 0.0], [-1.0, -2.0, -1.0]])
    filter_v = np.array([[1.0, 0.0, -1.0], [2.0, 0.0, -2.0], [1.0, 0.0, -1.0]])

    filter_u = np.stack([filter_u] * 3, axis=2)
    filter_v = np.stack([filter_v] * 3, axis=2)

    img = img.astype('float32')
    convolved = np.absolute(convolve(img, filter_u)) \
        + np.absolute(convolve(img, filter_v))

    energy_map = convolved.sum(axis=2)
    energy_map = energy_map + np.max(energy_map) * mask[:, :, 0]

    return energy_map


def getMinimumSeam(img, mask):
    """
    Get minimum seam of current image and mask
    :param img: input image
    :param mask: input mask
    :return: index of minimum seam
    """
    h, w, _ = img.shape

    energy_map = getEnergyMap(img, mask)
    backtrack = np.zeros_like(energy_map, dtype=np.int)
    for i in range(1, h):  # start from 1 to avoid leftmost edge
        for j in range(w):
            if j == 0:
                min_idx = np.argmin(energy_map[i - 1, j: j + 2])
                backtrack[i, j] = min_idx + j
                min_energy = energy_map[i - 1, min_idx + j]
            else:
                min_idx = np.argmin(energy_map[i - 1, j - 1: j + 2])
                backtrack[i, j] = min_idx + j - 1
                min_energy = energy_map[i - 1, min_idx + j - 1]
            energy_map[i, j] += min_energy

    return np.argmin(energy_map[-1]), backtrack


def carve(img, scale, mask):
    """
    Carve a seam on input image
    :param img: input image
    :param scale: ratio between input and output image
    :param mask: input mask
    :return carved_img: carved image
    :return mask: updated mask
    """
    h, w, _ = img.shape
    new_w = int(scale * w)

    carved_img = img.copy()
    for _ in range(w - new_w):
        h, w, _ = carved_img.shape
        seam, backtrack = getMinimumSeam(carved_img, mask)
        next_shape = np.ones((h, w), dtype=np.bool)
        for i in reversed(range(h)):
            next_shape[i, seam] = False
            seam = backtrack[i, seam]
        next_shape = np.stack([next_shape] * 3, axis=2)
        carved_img = carved_img[next_shape].reshape((h, w - 1, 3))
        mask = mask[next_shape].reshape((h, w - 1, 3))

    return carved_img, mask
