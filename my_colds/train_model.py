import sys
import glob
import os
import argparse

import numpy as np
import keras

from sklearn.model_selection import train_test_split

from my_colds.utils import get_class_label
from my_colds.detection.sore_throat.data import load_img_from_disk

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', action='store', dest='input_dir', required=True)
parser.add_argument('--output-dir', action='store', dest='output_dir', required=True)
parser.add_argument('--test-fraction', nargs='?', type=float, dest='test_fraction', default=.3)
parser.add_argument('--num-classes', nargs='?', type=int, dest='num_classes', default=2)
parser.add_argument('--batch-size', nargs='?', type=int, dest='batch_size', default=32)
parser.add_argument('--num-epochs', nargs='?', type=int, dest='epochs', default=5)
parser.add_argument('--loss', nargs='?', type=str, dest='loss', default='categorical_crossentropy')
parser.add_argument('--optimizer', nargs='?', type=str, dest='optimizer', default='adam')
parser.add_argument(
    '--final-img-dims',
    default=[420, 380, 3],
    nargs=3,
    metavar=('height', 'weight', 'depth'),
    type=int,
    help='specify a range',
    dest='img_dims'
)
parser.add_argument(
    '--metrics',
    default=['accuracy'],
    type=str,
    dest='metrics'
)

RANDOM_SEED = 42
ALPHA = 1


def run_training(x, y, model_params, fit_params, loss, optimizer, metrics, model_path):
    model = keras.applications.mobilenet.MobileNet(**model_params)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit(x, y, **fit_params)

    model.save(model_path)


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
    test_fraction = args.test_fraction
    num_classes = args.num_classes
    batch_size = args.batch_size
    epochs = args.epochs
    loss = args.loss
    optimizer = args.optimizer
    input_img_shape = args.img_dims
    metrics = args.metrics

    assert len(input_img_shape) == 3, f'--final-img-dims {img_dims} should have 3 dimensions'

    if not os.path.isdir(input_dir):
        raise EnvironmentError(f'--input-dir {input_dir} should exist')

    paths = glob.glob(f'{input_dir}/*')

    imgs = np.array(list(map(load_img_from_disk, paths)))
    labels = np.array(list(map(lambda path: int(get_class_label(path)), paths)))
    labels = keras.utils.to_categorical(labels, num_classes)

    assert len(paths) == labels.shape[0]

    X_train, X_val, y_train, y_val = train_test_split(
        imgs,
        labels,
        test_size=test_fraction,
        random_state=RANDOM_SEED
    )
    print('Training MobileNet for %s epochs with %d partitions of training data.' % (epochs, len(X_train)))

    model_params = {
        'input_tensor': keras.layers.Input(shape=input_img_shape),
        'alpha': ALPHA,
        'classes': num_classes,
        'weights': None,
    }
    # print(f'Model params {model_params}')

    fit_params = {
        'batch_size': batch_size,
        'epochs': epochs,
        'validation_data': (X_val, y_val),
        'shuffle': True,
        'verbose': 1,
    }
    # print(f'Fit params {fit_params}')

    os.makedirs(output_dir, exist_ok=True)
    model_path = f'{output_dir}/mobilenet_{epochs}_epochs.h5'
    run_training(X_train, y_train, model_params, fit_params, loss, optimizer, metrics, model_path)


if __name__ == '__main__':
    args = parser.parse_args()
    sys.exit(main(args))
