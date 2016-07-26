#!/usr/bin/env python3

"""Test/benchmark deep_dream.py."""

from pathlib import Path
import time

import click
from PIL import Image

import deep_dream as dd
import utils

utils.setup_traceback()


@click.command()
@click.option('--cpu-workers', default=0, help='The number of CPU workers to start.')
@click.option('--gpus', type=utils.IntList(), default='', help='The CUDA device IDs to use.')
@click.option('--max-tile-size', default=512, help='The maximum dimension of a tile.')
def main(cpu_workers=None, gpus=None, max_tile_size=None):
    pwd = Path(__file__).parent
    cnn = dd.CNN(dd.GOOGLENET_BVLC, cpu_workers=cpu_workers, gpus=gpus)
    input_img = Image.open(str(pwd/'kodim/img0022.jpg')).resize((1536, 1024), Image.LANCZOS)
    print('Input image classes:')
    for category in cnn.classify(input_img, 5):
        print('%.3f %s' % category)
    cnn.dream(input_img, 'inception_3a/3x3', min_size=129, n=1, max_tile_size=max_tile_size)
    time_0 = time.perf_counter()
    img = cnn.dream(input_img, 'inception_3a/3x3', min_size=129, n=10, step_size=1,
                    max_tile_size=max_tile_size)
    time_1 = time.perf_counter()
    print('Input image classes:')
    for category in cnn.classify(img, 5):
        print('%.3f %s' % category)
    print('Time taken: %.3f s' % (time_1-time_0))
    dd.to_image(img).save('test_output.png')
    print('Saved to test_output.png.')

if __name__ == '__main__':
    main()
