"""
Basic set of tools to make augmentations on image.
"""

import cv2
import numpy as np


class ImageProcessor:

    def __init__(self, shape: tuple):
        assert shape
        self.h = shape[0]
        self.w = shape[1]
        self.d = shape[2]

    def resize(self, img):
        if img.shape[0] / self.h >= img.shape[1] / self.w:
            img = cv2.resize(
                img,
                (int(self.h * img.shape[1] / img.shape[0]), self.h)
            )
        else:
            img = cv2.resize(
                img,
                (self.w, int(self.w * img.shape[0] / img.shape[1]))
            )
        return img

    def fil_blank(self, img_rgb):
        if img_rgb.shape[0] == self.h:
            int_resize_1 = img_rgb.shape[1]
            int_fill_1 = (self.w - int_resize_1) // 2
            int_fill_2 = self.w - int_resize_1 - int_fill_1
            numpy_fill_1 = np.zeros((self.h, int_fill_1, 3), dtype=np.uint8)
            numpy_fill_2 = np.zeros((self.h, int_fill_2, 3), dtype=np.uint8)
            img_filled_rgb = np.concatenate((numpy_fill_1, img_rgb, numpy_fill_2), axis=1)
        elif img_rgb.shape[1] == self.w:
            int_resize_0 = img_rgb.shape[0]
            int_fill_1 = (self.h - int_resize_0) // 2
            int_fill_2 = self.h - int_resize_0 - int_fill_1
            numpy_fill_1 = np.zeros((int_fill_1, self.w, 3), dtype=np.uint8)
            numpy_fill_2 = np.zeros((int_fill_2, self.w, 3), dtype=np.uint8)
            img_filled_rgb = np.concatenate((numpy_fill_1, img_rgb, numpy_fill_2), axis=0)
        else:
            raise ValueError
        return img_filled_rgb

    def __call__(self, img):
        img = self.resize(img)
        img = self.fil_blank(img)

        return img