from my_colds.detection.sore_throat.data.image import ImageDataLoader
from my_colds.detection.sore_throat.model.models import (
    PlaqueDetectorModelType,
    create_mobile_net_detector,
)


class PlaqueDetection:
    MODELS = {
        PlaqueDetectorModelType.MOBILE_NET: create_mobile_net_detector(),
    }

    def __init__(self, photo_url, from_web=True, with_augmentation=True):
        if with_augmentation:
            l = ImageDataLoader()
            if from_web:
                self.img = l.load_web_img(photo_url)
            else:
                self.img = l.load_local_img(photo_url)
        else:
            raise NotImplementedError

    def detect(self, model_type='mobile_net'):
        model_type = PlaqueDetectorModelType(model_type)
        not_plaque, plaque = PlaqueDetection.MODELS[model_type](self.img)[0]
        return not_plaque, plaque
