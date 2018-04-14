import sys
import os
import argparse

import keras

# TODO: remove so as obsolete code!

parser = argparse.ArgumentParser()
parser.add_argument('--input-dir', action='store', dest='input_dir', required=True)
parser.add_argument('--output-dir', action='store', dest='output_dir', required=True)
parser.add_argument('--num-classes', nargs='?', type=int, dest='num_classes', default=2)
parser.add_argument('--batch-size', nargs='?', type=int, dest='batch_size', default=32)
parser.add_argument('--num-epochs', nargs='?', type=int, dest='epochs', default=5)
parser.add_argument('--loss', nargs='?', type=str, dest='loss', default='categorical_crossentropy')
parser.add_argument('--optimizer', nargs='?', type=str, dest='optimizer', default='adam')
parser.add_argument(
    '--final-img-dims',
    default=[256, 256, 3],
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

ALPHA = 1


def build_augmentation_pipeline(path):
    p = Augmentor.Pipeline(path)
    p.flip_top_bottom(probability=0.1)
    p.rotate(probability=0.3, max_left_rotation=5, max_right_rotation=5)
    return p


def run_training(train_data_generator, model_params, fit_params, loss, optimizer, metrics, model_path):
    model = keras.applications.mobilenet.MobileNet(**model_params)

    model.compile(loss=loss, optimizer=optimizer, metrics=metrics)

    model.fit_generator(train_data_generator, **fit_params)

    model.save(model_path)


def main(args):
    input_dir = args.input_dir
    output_dir = args.output_dir
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

    train_part_dir = f'{input_dir}/train'
    valid_part_dir = f'{input_dir}/valid'

    train_p = build_augmentation_pipeline(train_part_dir)
    train_g = train_p.keras_generator(batch_size=batch_size)
    valid_p = build_augmentation_pipeline(valid_part_dir)
    valid_g = valid_p.keras_generator(batch_size=batch_size)
    print(f'Training MobileNet for {epochs} epochs.')

    model_params = {
        'input_tensor': keras.layers.Input(shape=input_img_shape),
        'alpha': ALPHA,
        'classes': num_classes,
        'weights': None,
    }

    fit_params = {
        'steps_per_epoch': len(train_p.augmentor_images) / batch_size,
        'epochs': epochs,
        'validation_steps': len(valid_p.augmentor_images) / batch_size,
        'validation_data': valid_g,
        'shuffle': True,
        'verbose': 1,
    }

    os.makedirs(output_dir, exist_ok=True)
    model_path = f'{output_dir}/mobilenet_{epochs}_epochs.h5'
    run_training(train_g, model_params, fit_params, loss, optimizer, metrics, model_path)


if __name__ == '__main__':
    args = parser.parse_args()
    sys.exit(main(args))
