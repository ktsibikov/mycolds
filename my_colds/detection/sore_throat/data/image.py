"""Basic set of tools to load images and do some augmentations"""
import requests
from PIL import Image
from io import BytesIO
import numpy as np
from keras.preprocessing.image import (
    load_img,
    ImageDataGenerator,
    img_to_array,
)

try:
    from PIL import Image as pil_image
except ImportError:
    pil_image = None

if pil_image is not None:
    _PIL_INTERPOLATION_METHODS = {
        'nearest': pil_image.NEAREST,
    }

TARGET_SIZE = 224, 224


class ImageDataLoader:
    def __init__(
        self,
        rotation_range=40,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=20.,
        horizontal_flip=True,
        fill_mode='nearest'
    ):
        self.interpolation = fill_mode
        self.data_gen = ImageDataGenerator(
            rotation_range=rotation_range,
            width_shift_range=width_shift_range,
            height_shift_range=height_shift_range,
            shear_range=shear_range,
            zoom_range=zoom_range,
            channel_shift_range=channel_shift_range,
            horizontal_flip=horizontal_flip,
            fill_mode=fill_mode
        )

    def reshape_img(self, img, target_size):
        if target_size is not None:
            width_height_tuple = (target_size[1], target_size[0])
            if img.size != width_height_tuple:
                if self.interpolation not in _PIL_INTERPOLATION_METHODS:
                    raise ValueError(
                        'Invalid interpolation method {} specified. Supported '
                        'methods are {}'.format(
                            self.interpolation,
                            ', '.join(_PIL_INTERPOLATION_METHODS.keys())
                        )
                    )
                resample = _PIL_INTERPOLATION_METHODS[self.interpolation]
                img = img.resize(width_height_tuple, resample)
        return img

    def process(self, img):
        img_tensor = img_to_array(img)
        augmented = self.data_gen.random_transform(img_tensor)
        extended = np.expand_dims(augmented, axis=0)
        normalized = extended / 255.
        return normalized

    def load_web_img(self, url, target_size=TARGET_SIZE):
        response = requests.get(url)
        img = Image.open(BytesIO(response.content))
        img = self.reshape_img(img, target_size)
        augmented = self.process(img)
        return augmented

    def load_local_img(self, path, target_size=TARGET_SIZE):
        img = load_img(path, target_size=target_size, interpolation=self.interpolation)
        augmented = self.process(img)
        return augmented
