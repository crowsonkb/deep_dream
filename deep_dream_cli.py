#!/usr/bin/env python3

"""CLI interface to deep_dream."""

import math
import sys
from types import SimpleNamespace

import click
from PIL import Image

import deep_dream as dd
import utils

utils.setup_traceback()


@click.command()
@click.argument('in_file', type=click.Path(exists=True))
@click.argument('out_file', default='out.png')
@click.option('--cpu-workers', type=int, default=0, help='The number of CPU workers to start.')
@click.option('--gpus', type=utils.List(int, 'integer'), default='',
              help='The CUDA device IDs to use.')
@click.option('--guide-image', type=str, default=None, help='The guide image to use.')
@click.option('--layers', type=utils.List(str, 're'), default='',
              help='The network layers to target.')
@click.option('--max-input-size', type=int, nargs=2, default=None,
              help='Rescale the input image to fit into this size.')
@click.option('--max-tile-size', default=512, help='The maximum dimension of a tile.')
@click.option('--min-size', default=128,
              help='Don\'t use scales where the small edge of the image is below this.')
# TODO: allow custom models
# TODO: dynamically fill models list
@click.option('--model', default='GOOGLENET_BVLC', help='The model to use. Valid values: '
              'GOOGLENET_BVLC, GOOGLENET_PLACES205, GOOGLENET_PLACES365, RESNET_50.')
@click.option('--n', default=10, help='The number of iterations per scale.')
@click.option('--per-octave', default=2, help='The number of scales per octave.')
@click.option('--smoothing', default=0.0, help='The per-iteration smoothing factor. Try 0.02-0.1.')
@click.option('--step-size', default=1.5, help='The strength of each iteration.')
@click.option('--step-size-fac', default=1.0,
              help='The factor to multiply step_size by each octave.')
@click.option('--tv-weight', type=float, default=None, help='The per-scale denoising weight. '
              'Higher values smooth the image less. Try 25-250.')
def main(**kwargs):
    """CLI interface to deep_dream."""
    print('Arguments:')
    for arg in sorted(kwargs.items()):
        print('    %s: %s' % arg)
    print()
    args = SimpleNamespace(**kwargs)

    args.model = args.model.upper()
    model = getattr(dd, args.model)
    assert model.__class__ == dd.CNNData
    assert args.cpu_workers >= 0
    cnn = dd.CNN(model, cpu_workers=args.cpu_workers, gpus=args.gpus)
    layers = []
    for expr in args.layers:
        try:
            layers.extend(cnn.layers(expr))
        except KeyError:
            layers = None
            break
    if not layers:
        print('List of valid layers:')
        for layer in cnn.layers():
            print('    ' + layer)
        sys.exit(1)

    in_img = Image.open(args.in_file)

    if args.max_input_size:
        assert len(args.max_input_size) == 2
        max_w, max_h = args.max_input_size
        assert max_w >= 32 and max_h >= 32
        w, h = in_img.size
        fac = 1
        if w > max_w:
            fac = max_w / w
        if h > max_h:
            fac = min(fac, max_h / h)
        w, h = round(w*fac), round(h*fac)
        in_img = in_img.resize((w, h), Image.LANCZOS)

    if args.guide_image is None:
        weights = cnn.prepare_layer_list(layers)
        print('Layer weights:')
        for weight in reversed(weights.items()):
            print('    %s: %g' % weight)
    else:
        guide_img = Image.open(args.guide_image)
        weights = cnn.prepare_guide_weights(guide_img, layers)
        print('Layers:')
        for layer in reversed(weights.keys()):
            print('    %s' % layer)
    print()

    print('Input image size: %dx%d\n' % in_img.size)

    def fn(size):
        h, w = size
        x = min(w, h)
        ss = args.step_size * args.step_size_fac ** (math.log2(x) - math.log2(args.min_size))
        print('Scale: %dx%d, step_size=%0.2f' % (w, h, ss))
        return {'step_size': ss}

    img = cnn.dream(in_img, weights, fn=fn, max_tile_size=args.max_tile_size,
                    min_size=args.min_size, n=args.n, per_octave=args.per_octave,
                    smoothing=args.smoothing, step_size=args.step_size, tv_weight=args.tv_weight)

    save_args = {}
    out_type = args.out_file.rpartition('.')[2].lower()
    if out_type == 'jpg' or out_type == 'jpeg':
        save_args['quality'] = 95  # TODO: make configurable
    dd.to_image(img).save(args.out_file, **save_args)
    print('Saved to %s.' % args.out_file)

if __name__ == '__main__':
    main()
