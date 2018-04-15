import sys
import os
import argparse

from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Input
from keras.applications.mobilenet import MobileNet
from keras import backend as K

parser = argparse.ArgumentParser()
parser.add_argument('--train-data', action='store', dest='train_data', required=True)
parser.add_argument('--valid-data', action='store', dest='valid_data', required=True)
parser.add_argument('--model-path', action='store', dest='model_path', required=True)
parser.add_argument(
    '--model-weights-path',
    action='store',
    dest='model_weights_path',
    required=True
)
parser.add_argument('--num-classes', nargs='?', type=int, dest='num_classes', default=2)
parser.add_argument('--batch-size', nargs='?', type=int, dest='batch_size', default=8)
parser.add_argument('--train-samples', nargs='?', type=int, dest='train_samples', required=True)
parser.add_argument('--valid-samples', nargs='?', type=int, dest='valid_samples', required=True)
parser.add_argument('--num-epochs', nargs='?', type=int, dest='epochs', default=1)
parser.add_argument('--loss', nargs='?', type=str, dest='loss', default='categorical_crossentropy')
parser.add_argument('--optimizer', nargs='?', type=str, dest='optimizer', default='adam')
parser.add_argument(
    '--target-size',
    default=[224, 224],
    nargs=2,
    metavar=('height', 'weight'),
    type=int,
    help='Target image size',
    dest='target_size'
)

ALPHA = 1


def main(args):
    train_data = args.train_data
    valid_data = args.valid_data
    nb_train_samples = args.train_samples
    nb_validation_samples = args.valid_samples
    model_path = args.model_path
    model_weights_path = args.model_weights_path
    num_classes = args.num_classes
    batch_size = args.batch_size
    epochs = args.epochs
    loss = args.loss
    optimizer = args.optimizer
    img_width, img_height = args.target_size

    if not os.path.isdir(train_data):
        raise EnvironmentError(f'--train-data {train_data} should exist')

    if not os.path.isdir(valid_data):
        raise EnvironmentError(f'--valid-data {valid_data} should exist')

    if K.image_data_format() == 'channels_first':
        input_shape = (3, img_width, img_height)
    else:
        input_shape = (img_width, img_height, 3)

    model_params = {
        'input_tensor': Input(shape=input_shape),
        'classes': num_classes,
        'weights': None,
    }

    print(
        f'Start training mobile net for {epochs} epochs.',
        f'{nb_train_samples} train samples, {nb_validation_samples} valid samples'
    )

    model = MobileNet(**model_params)
    model.compile(
        loss=loss,
        optimizer=optimizer,
        metrics=['accuracy']
    )

    train_datagen = ImageDataGenerator(rescale=1. / 255)
    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        train_data,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
    )

    validation_generator = test_datagen.flow_from_directory(
        valid_data,
        target_size=(img_width, img_height),
        batch_size=batch_size,
        class_mode='categorical',
    )

    model.fit_generator(
        train_generator,
        steps_per_epoch=nb_train_samples // batch_size,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=nb_validation_samples // batch_size,
        verbose=1
    )

    model.save(model_path)
    model.save_weights(model_weights_path)

    print('Model saved.')
    return 0


if __name__ == '__main__':
    args = parser.parse_args()
    sys.exit(main(args))
