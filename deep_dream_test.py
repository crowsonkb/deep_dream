"""Test/benchmark deep_dream.py."""

import argparse
from pathlib import Path
import time

from PIL import Image

import deep_dream as dd


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--gpu', type=int, help='the CUDA device ID to use')
    parser.add_argument('--max-tile-size', type=int, default=512,
                        help='the maximum dimension of a tile')
    parser.add_argument('--save', action='store_true', help='save result to output.png')
    args = parser.parse_args()

    pwd = Path(__file__).parent
    cnn = dd.CNN(dd.GOOGLENET_BVLC, gpu=args.gpu)
    input_img = Image.open(str(pwd/'kodim/img0022.jpg')).resize((768, 512), Image.LANCZOS)
    cnn.dream(input_img, 'inception_3a/3x3', scale=4, n=1, max_tile_size=args.max_tile_size)
    t1 = time.perf_counter()
    img = cnn.dream(input_img, 'inception_3a/3x3', scale=4, n=10, max_tile_size=args.max_tile_size)
    t2 = time.perf_counter()
    print('Time taken: %.3f s' % (t2-t1))
    if args.save:
        dd.to_image(img).save('output.png')

if __name__ == '__main__':
    main()
