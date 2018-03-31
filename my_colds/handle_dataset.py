"""
The script take all images from passed directory,
process them and store final images to the passed
output directory.
"""
import sys
import os
import glob
import argparse

import cv2

from my_colds.classification.sore_throat.data import ImageProcessor


def get_class_lable(file_name: str) -> str:
    return file_name[5]


def main(input_dir: str, output_dir: str, img_shape: tuple) -> int:
    train_files_pattern = f'{input_dir}/*'
    globbed = glob.glob(train_files_pattern)

    assert len(globbed) > 0, f'--input-dir {input_dir} shouldn\'t be empty'

    processed = []
    process = ImageProcessor(img_shape)

    for path in globbed:
        file_name = path.rpartition('/')[2]
        label = get_class_lable(file_name)
        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        images = [process(img)]
        processed.extend(zip([label] * len(images), images))

    os.makedirs(output_dir, exist_ok=True)

    for idx, (label, img) in enumerate(processed, 1):
        path = f'{output_dir}/{label}_{idx}.jpg'
        cv2.imwrite(path, img)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Process raw image dataset and store converted data.')
    parser.add_argument('--input-dir', action='store', dest='input_dir', required=True)
    parser.add_argument('--output-dir', action='store', dest='output_dir', required=True)
    parser.add_argument('--final-img-dims', type=int, nargs='+', dest='img_dims', required=True)
    args = parser.parse_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    img_dims = tuple(args.img_dims)

    assert len(img_dims) == 3, f'--final-img-dims {img_dims} should have 3 dimensions'

    if not os.path.isdir(input_dir):
        raise EnvironmentError(f'--input-dir {input_dir} should exist')

    sys.exit(main(input_dir, output_dir, img_dims))