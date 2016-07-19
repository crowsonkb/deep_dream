#!/usr/bin/env python3

"""Test/benchmark deep_dream.py."""

import argparse
from pathlib import Path
import time

from PIL import Image

import deep_dream as dd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cpu-workers', type=int, default=0,
                        help='the number of CPU workers to start')
    parser.add_argument('--gpus', type=int, nargs='+', default=[],
                        help='the CUDA device IDs to use')
    parser.add_argument('--max-tile-size', type=int, default=512,
                        help='the maximum dimension of a tile')
    parser.add_argument('--save', action='store_true', help='save result to output.png')
    args = parser.parse_args()

    pwd = Path(__file__).parent
    if not args.gpus and not args.cpu_workers:
        args.cpu_workers = 1
    cnn = dd.CNN(dd.GOOGLENET_BVLC, cpu_workers=args.cpu_workers, gpus=args.gpus)
    input_img = Image.open(str(pwd/'kodim/img0022.jpg')).resize((768, 512), Image.LANCZOS)
    print('Input image classes:')
    for c in cnn.classify(input_img, 5):
        print('%.3f %s' % c)
    cnn.dream(input_img, 'inception_3a/3x3', min_size=129, n=1, max_tile_size=args.max_tile_size)
    t1 = time.perf_counter()
    img = cnn.dream(input_img, 'inception_3a/3x3', min_size=129, n=10, max_tile_size=args.max_tile_size)
    t2 = time.perf_counter()
    print('Input image classes:')
    for c in cnn.classify(img, 5):
        print('%.3f %s' % c)
    print('Time taken: %.3f s' % (t2-t1))
    if args.save:
        dd.to_image(img).save('output.png')

if __name__ == '__main__':
    main()
