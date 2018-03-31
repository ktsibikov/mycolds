"""
The script take all images from passed directory,
process them and store final images to the passed
output directory.
"""
import sys
import os
import glob
import argparse

from my_colds.utils import get_class_label
from my_colds.detection.sore_throat.data import (
    ImageProcessor,
    load_img_from_disk,
    save_img,
)


def main(in_dir: str, out_dir: str, img_shape: tuple) -> int:
    train_files_pattern = f'{in_dir}/*'
    globed = glob.glob(train_files_pattern)

    assert len(globed) > 0, f'--input-dir {in_dir} should not be empty'

    processed = []
    process = ImageProcessor(img_shape)

    for path in globed:
        label = get_class_label(path)
        img = load_img_from_disk(path)
        images = [process(img)]
        processed.extend(zip([label] * len(images), images))

    os.makedirs(out_dir, exist_ok=True)

    for idx, (label, img) in enumerate(processed, 1):
        path = f'{out_dir}/{label}_{idx}.jpg'
        save_img(path, img)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process raw image data set and store converted data.')
    parser.add_argument('--input-dir', action='store', dest='input_dir', required=True)
    parser.add_argument('--output-dir', action='store', dest='output_dir', required=True)
    parser.add_argument(
        '--final-img-dims',
        default=[420, 380, 3],
        nargs=3,
        metavar=('height', 'weight', 'depth'),
        type=int,
        help='specify a range',
        dest='img_dims'
    )
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    img_dims = tuple(args.img_dims)

    assert len(img_dims) == 3, f'--final-img-dims {img_dims} should have 3 dimensions'

    if not os.path.isdir(input_dir):
        raise EnvironmentError(f'--input-dir {input_dir} should exist')

    sys.exit(main(input_dir, output_dir, img_dims))
