import os
import cv2
import argparse
import numpy as np
import skimage.io as sio

from scipy import signal
from utils import getBaseLayer, getAvgMask, carve


class ObjectRemover(object):
    def __init__(self, path_to_img, kernel_size=30, scale=0.5, rotate=False):
        self.img_path = path_to_img
        self.kernel_size = kernel_size
        self.scale = scale
        self.rotate = rotate

        self.img = sio.imread(self.img_path)
        self.img_shape = self.img.shape

    def run(self):
        base = getBaseLayer(self.img, kernel_size=100, sigma=3)

        kernel = np.ones((self.kernel_size, self.kernel_size))
        kernel[:, 0: self.kernel_size // 2] = -1

        h, w, c = self.img_shape
        base_convolved = np.zeros(base.shape)
        for idx in range(c):
            base_convolved[:, :, idx] = signal.convolve2d(
                    base[:, :, idx], kernel, boundary='symm', mode='same')

        base_convolved = np.abs(base_convolved)
        base_convolved = base_convolved - np.mean(base_convolved)
        base_convolved = np.clip(base_convolved / 255, 0, 1).astype('float')

        base_diff = np.abs(base - np.median(base))
        base_diff = base_diff - np.mean(base_diff)
        base_diff = np.clip(base_diff, 0, 255).astype('float') / 255

        base_convolved = base_convolved * (1 - np.mean(base_convolved)) \
            + base_diff * (1 - np.mean(base_diff))
        base_convolved = np.clip(base_convolved, 0, 1).astype('float')

        base_convolved = getAvgMask(base_diff, kernel_h=self.kernel_size // 2,
                                    kernel_w=self.kernel_size // 2)
        base_convolved = base_convolved - np.min(base_convolved)

        base_diff = getAvgMask(base_diff, kernel_h=self.kernel_size // 2,
                               kernel_w=self.kernel_size // 2)
        base_diff = base_diff - np.min(base_diff)

        mask_diff = 1 - base_diff / np.max(base_diff)

        mask = np.zeros(self.img_shape)
        for idx in range(c):
            mask[:, :, idx] = mask_diff

        if self.rotate:
            rotated_img = np.rot90(self.img, 1, (0, 1))
            rotated_mask = np.rot90(mask, 1, (0, 1))
            output, _ = carve(rotated_img, self.scale, rotated_mask)
            output = np.rot90(output, 3, (0, 1))
        else:
            output, _ = carve(self.img, self.scale, mask)

        return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-img', '--img_path', required=True,
                        help='path to target image')
    parser.add_argument('-s', '--scale', type=float, default=0.7,
                        help='scale, ex: 0.7')
    parser.add_argument('-r', '--rotate', default=0,
                        help='set to 1 if rotate')
    args = parser.parse_args()

    img_path = args.img_path
    scale = args.scale
    rotate = args.rotate

    object_remover = ObjectRemover(path_to_img=img_path,
                                   scale=scale,
                                   rotate=rotate)

    out = object_remover.run()
    out = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
    out_path = os.path.join(
        os.path.dirname(img_path),
        os.path.splitext(os.path.basename(img_path))[0] + '_output.jpg')
    cv2.imwrite(out_path, out)
