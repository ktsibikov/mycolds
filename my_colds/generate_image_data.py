"""
The script generates augmented images. Receive path to the
folder with data. Directory structure must be the next:
--train
    --cats
        0001.jpeg
        0002.jpeg
        ...
    --dogs
        0001.jpeg
        ....
Augmented images will be stored in the folder by given output_path.
"""
import sys
import os
import argparse

from keras.preprocessing.image import ImageDataGenerator


def main(in_dir: str, out_dir: str, target_size: tuple, img_counts) -> int:
    gen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.15,
        height_shift_range=0.15,
        shear_range=0.2,
        zoom_range=0.2,
        channel_shift_range=20.,
        horizontal_flip=True,
        fill_mode='nearest'
    )

    labels = {}
    for idx, batch in enumerate(gen.flow_from_directory(
        in_dir,
        target_size=target_size,
        class_mode='binary',
        shuffle=True,
        batch_size=1,
        save_to_dir=out_dir,
        save_prefix='N',
        save_format='jpeg',
    ), 1):
        for l in map(int, batch[1]):
            labels.setdefault(l, 0)
            labels[l] += 1
        if idx >= img_counts:
            break

    print('Count of generated images by classes:\n', labels)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', action='store', dest='input_dir', required=True)
    parser.add_argument('--output-dir', action='store', dest='output_dir', required=True)
    parser.add_argument(
        '--final-image-count',
        type=int,
        help='The number of images to generate',
        required=True
    )
    parser.add_argument(
        '--target-size',
        default=[256, 256],
        nargs=2,
        metavar=('height', 'weight'),
        type=int,
        help='Size of final images.',
        dest='target_size'
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    img_counts = args.final_image_count
    target_size = tuple(args.target_size)

    if not os.path.isdir(input_dir):
        raise EnvironmentError(f'--input-dir {input_dir} should exist')

    sys.exit(main(input_dir, output_dir, target_size, img_counts))
