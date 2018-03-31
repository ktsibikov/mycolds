"""
The script take all images from passed directory,
process them and store final images to the passed
output directory.
"""
import sys
import os
import glob
import argparse

from sklearn.model_selection import train_test_split

from my_colds.utils import get_class_label
from my_colds.detection.sore_throat.data import (
    ImageProcessor,
    load_img_from_disk,
    save_img,
)

RANDOM_SEED = 42


def main(in_dir: str, out_dir: str, img_shape: tuple, test_size: float) -> int:
    train_files_pattern = f'{in_dir}/*'
    globed = glob.glob(train_files_pattern)

    assert len(globed) > 0, f'--input-dir {in_dir} should not be empty'

    images = []
    labels = []
    process = ImageProcessor(img_shape)

    for path in globed:
        label = get_class_label(path)
        img = load_img_from_disk(path)
        images.append(process(img))
        labels.append(label)

    X_train, X_val, y_train, y_val = train_test_split(
        images,
        labels,
        test_size=test_size,
        random_state=RANDOM_SEED
    )

    train_part_dir = f'{out_dir}/train/'
    valid_part_dir = f'{out_dir}/valid/'

    for idx, (img, label) in enumerate(zip(X_train, y_train), 1):
        path = f'{train_part_dir}/{label}/{idx}.jpg'
        os.makedirs(path.rpartition('/')[0], exist_ok=True)
        save_img(path, img)

    for idx, (img, label) in enumerate(zip(X_val, y_val), 1):
        path = f'{valid_part_dir}/{label}/{idx}.jpg'
        os.makedirs(path.rpartition('/')[0], exist_ok=True)
        save_img(path, img)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process raw image data set and store converted data.')
    parser.add_argument('--input-dir', action='store', dest='input_dir', required=True)
    parser.add_argument('--output-dir', action='store', dest='output_dir', required=True)
    parser.add_argument('--test-fraction', nargs='?', type=float, dest='test_fraction', default=.3)
    parser.add_argument(
        '--final-img-dims',
        default=[256, 256, 3],
        nargs=3,
        metavar=('height', 'weight', 'depth'),
        type=int,
        help='specify a range',
        dest='img_dims'
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    test_fraction = args.test_fraction
    img_dims = tuple(args.img_dims)

    assert len(img_dims) == 3, f'--final-img-dims {img_dims} should have 3 dimensions'

    if not os.path.isdir(input_dir):
        raise EnvironmentError(f'--input-dir {input_dir} should exist')

    sys.exit(main(input_dir, output_dir, img_dims, test_fraction))
