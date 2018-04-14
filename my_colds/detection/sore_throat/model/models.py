from enum import Enum

from keras.models import load_model
from keras.utils.generic_utils import CustomObjectScope
from keras.applications.mobilenet import (
    relu6,
    DepthwiseConv2D,
)

from config import CONFIG


class PlaqueDetectorModelType(Enum):
    MOBILE_NET = 'mobile_net'


class MobileNetDetector:
    def __init__(self, model_path, model_weights_path):
        with CustomObjectScope(
                {
                    'relu6': relu6,
                    'DepthwiseConv2D': DepthwiseConv2D,
                }
        ):
            self.model = load_model(model_path)
        self.model.load_weights(model_weights_path)

    def __call__(self, img_tensor):
        return self.model.predict(img_tensor)


def create_mobile_net_detector():
    detection_config = CONFIG['detection']
    model = MobileNetDetector(
        detection_config['model_path'],
        detection_config['model_weights_path']
    )
    return model
