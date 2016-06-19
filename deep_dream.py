"""Deep Dreaming using Caffe and Google's Inception convolutional neural network."""

# pylint: disable=invalid-name

from collections import namedtuple, OrderedDict
import os
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage
from tqdm import tqdm

os.environ['GLOG_minloglevel'] = '1'
import caffe  # pylint: disable=wrong-import-position

EPS = np.finfo(np.float32).eps
SOFTEN = np.float32([[[1, 2, 1], [2, 20, 2], [1, 2, 1]]])/32

CNNData = namedtuple('CNNData', 'deploy model mean categories')
CNNData.__new__.__defaults__ = (None,)  # Make categories optional.

_BASE_DIR = Path(__file__).parent
GOOGLENET_BVLC = CNNData(
    _BASE_DIR/'bvlc_googlenet/deploy.prototxt',
    _BASE_DIR/'bvlc_googlenet/bvlc_googlenet.caffemodel',
    (104, 117, 123),
    categories=_BASE_DIR/'bvlc_googlenet/categories.txt')
GOOGLENET_PLACES205 = CNNData(
    _BASE_DIR/'googlenet_places205/deploy_places205.prototxt',
    _BASE_DIR/'googlenet_places205/googlenet_places205_train_iter_2400000.caffemodel',
    (104.051, 112.514, 116.676))  # TODO: find the actual Places205 mean
GOOGLENET_PLACES365 = CNNData(
    _BASE_DIR/'googlenet_places365/deploy_googlenet_places365.prototxt',
    _BASE_DIR/'googlenet_places365/googlenet_places365.caffemodel',
    (104.051, 112.514, 116.676),
    categories=_BASE_DIR/'googlenet_places365/categories_places365.txt')


def to_image(arr):
    """Clips the values in a float32 ndarray to 0-255 and converts it to a PIL image."""
    return Image.fromarray(np.uint8(np.clip(np.round(arr), 0, 255)))


def _resize(arr, size, method=Image.BICUBIC):
    h, w = size
    arr = np.float32(arr)
    if arr.ndim == 3:
        planes = [arr[i, :, :] for i in range(arr.shape[0])]
    else:
        raise TypeError('Only 3D CxHxW arrays are supported')
    imgs = [Image.fromarray(plane) for plane in planes]
    imgs_resized = [img.resize((w, h), method) for img in imgs]
    return np.stack([np.array(img) for img in imgs_resized])


class ShapeError(Exception):
    """Raised by CNN when an invalid layer shape is requested which would otherwise crash Caffe."""
    def __str__(self):
        return 'bad shape %s at scale=%d' % self.args


class _LayerIndexer:
    def __init__(self, net, attr):
        self.net, self.attr = net, attr

    def __getitem__(self, key):
        return getattr(self.net.blobs[key], self.attr)[0]

    def __setitem__(self, key, value):
        getattr(self.net.blobs[key], self.attr)[0] = value


class _ChannelVecIndexer:
    def __init__(self, net):
        self.net = net

    def __getitem__(self, key):
        return np.zeros((self.net.blobs[key].data.shape[1], 1, 1), dtype=np.float32)


class CNN:
    """Represents an instance of a Caffe convolutional neural network."""

    def __init__(self, cnndata, gpu=None):
        """Initializes a CNN.

        Args:
            gpu (Optional[int]): If present, Caffe will use this GPU device number. On a typical
                system with one GPU, it should be 0. If not present Caffe will use the CPU.
        """
        self.start = 'data'
        self.net = caffe.Classifier(str(cnndata.deploy), str(cnndata.model),
                                    mean=np.float32(cnndata.mean), channel_swap=(2, 1, 0))
        self.categories = None
        if cnndata.categories is not None:
            self.categories = open(str(cnndata.categories)).read().splitlines()
        self.data = _LayerIndexer(self.net, 'data')
        self.diff = _LayerIndexer(self.net, 'diff')
        self.vec = _ChannelVecIndexer(self.net)
        self.img = np.zeros_like(self.data[self.start])
        self.total_px = 0
        self.progress_bar = None
        if gpu is not None:
            caffe.set_device(gpu)
            caffe.set_mode_gpu()
        else:
            caffe.set_mode_cpu()

    def _preprocess(self, img):
        return np.rollaxis(np.float32(img), 2)[::-1] - self.net.transformer.mean['data']

    def _deprocess(self, img):
        return np.dstack((img + self.net.transformer.mean['data'])[::-1])

    def prob(self, input_img, max_tile_size=512):
        """Classifies the input image and returns a probability distribution over possible classes.

        Args:
            input_img: The image to process (PIL images or Numpy arrays are accepted).
            max_tile_size: Does not allow the image dimension to exceed this.

        Returns:
            An ndarray containing a probability distribution over categories."""
        input_arr = self._preprocess(np.float32(input_img))
        if input_arr.shape[1] > max_tile_size or input_arr.shape[2] > max_tile_size:
            h = min(max_tile_size, input_arr.shape[1])
            w = min(max_tile_size, input_arr.shape[2])
            input_arr = _resize(input_arr, (h, w))
        self.net.blobs[self.start].reshape(1, 3, h, w)
        self.data[self.start] = input_arr
        end = self.layers()[-1]
        self.net.forward(end=end)
        return self.data[end].copy()

    def classify(self, input_img, n=1, **kwargs):
        """Classifies the input image and returns the n most probable categories.

        Args:
            input_img: The image to process (PIL images or Numpy arrays are accepted).
            n: The n most probable categories to return.
            max_tile_size: Does not allow the image dimension to exceed this.

        Returns:
            A list containing the n most probable categories."""
        indices = self.prob(input_img, **kwargs).argsort()[::-1][:n]
        if self.categories is None:
            return indices
        return [self.categories[i] for i in indices]

    def _grad_tiled(self, layers, progress=False, max_tile_size=512):
        # pylint: disable=too-many-locals
        if progress:
            if not self.progress_bar:
                self.progress_bar = tqdm(
                    total=self.total_px, unit='pix', unit_scale=True, ncols=80, smoothing=0.1)

        h, w = self.img.shape[1:]  # Height and width of input image
        ny, nx = (h-1)//max_tile_size+1, (w-1)//max_tile_size+1  # Number of tiles per dimension
        g = np.zeros_like(self.img)
        for y in range(ny):
            th = h//ny
            if y == ny-1:
                th += h - th*ny
            for x in range(nx):
                tw = w//nx
                if x == nx-1:
                    tw += w - tw*nx
                self.net.blobs[self.start].reshape(1, 3, th, tw)
                sy, sx = h//ny*y, w//nx*x
                self.data[self.start] = self.img[:, sy:sy+th, sx:sx+tw]

                for layer in layers.keys():
                    self.diff[layer] = 0
                self.net.forward(end=next(iter(layers.keys())))
                layers_list = list(layers.keys())
                for i, layer in enumerate(layers_list):
                    self.diff[layer] += self.data[layer] * layers[layer]
                    if i+1 == len(layers):
                        self.net.backward(start=layer)
                    else:
                        self.net.backward(start=layer, end=layers_list[i+1])

                g[:, sy:sy+th, sx:sx+tw] = self.diff[self.start]

                if progress:
                    self.progress_bar.update(th*tw)
        return g

    def _step(self, n=1, step_size=1.5, jitter=32, seed=0, **kwargs):
        np.random.seed(self.img.size + seed)
        for _ in range(n):
            x, y = np.random.randint(-jitter, jitter+1, 2)
            self.img = np.roll(np.roll(self.img, x, 2), y, 1)
            g = self._grad_tiled(**kwargs)
            self.img += step_size * g / (np.mean(np.abs(g)) + EPS)
            self.img = np.roll(np.roll(self.img, -x, 2), -y, 1)

    def _octave_detail(self, base, scale=4, n=10, per_octave=2, kernel=SOFTEN, **kwargs):
        if base.shape[1] < 32 or base.shape[2] < 32:
            raise ShapeError(base.shape, scale)
        factor = 2**(1/per_octave)
        detail = np.zeros_like(base, dtype=np.float32)
        self.total_px += base.shape[1] * base.shape[2] * n
        if scale != 1:
            hf, wf = np.int32(np.round(np.array(base.shape)[-2:]/factor))
            smaller_base = _resize(base, (hf, wf))
            smaller_detail = self._octave_detail(smaller_base, scale-1, n, per_octave, **kwargs)
            detail = _resize(smaller_detail, base.shape[-2:])
        self.img = base + detail
        self._step(n, **kwargs)
        detail = self.img - base
        if kernel is not None:
            detail = ndimage.convolve(detail, kernel)
        return detail

    def layers(self):
        """Returns a list of layer names, suitable for the 'end' argument of dream()."""
        layers = []
        for i, layer in enumerate(self.net.blobs.keys()):
            if i == 0 or layer.partition('_split_')[1]:
                continue
            layers.append(layer)
        return layers

    def dream(self, input_img, layers, progress=True, return_ndarray=False, **kwargs):
        """Runs the Deep Dream multiscale gradient ascent algorithm on the input image.

        Args:
            input_img: The image to process (PIL images or Numpy arrays are accepted)
            layers (dict): The layers/weights to use as an objective function for gradient ascent.
            progress (Optional[bool]): Display a progress bar while computing.
            scale (Optional[int]): The number of scales to process.
            per_octave (Optional[int]): Determines the difference between each scale; for instance,
                the default of 2 means that a 1000x1000 input image will get processed as 707x707
                and 500x500.
            n (Optional[int]): The number of gradient ascent steps per scale. Defaults to 10.
            step_size (Optional[float]): The strength of each individual gradient ascent step.
                Specifically, each step will change the image's pixel values by a median of
                step_size.
            max_tile_size (Optional[int]): Defaults to 512, suitable for a GPU with 2 GB RAM.
                Higher values perform better; if Caffe runs out of GPU memory and crashes then it
                should be lowered.

        Returns:
            The processed image, as a PIL image.

            If ndarray is true, returns the unclipped processed image as a float32 ndarray which
            has a valid range of 0-255 but which may contain components that are less than 0 or
            greater than 255. Both the PIL image and the ndarray are valid inputs to dream().
            deep_dream.to_image() can be used to convert the ndarray to a PIL image.
        """
        if isinstance(layers, str):
            layers = [layers]
        if isinstance(layers, list):
            layers = {layer: 1 for layer in layers}
        _layers = OrderedDict()
        for layer in reversed(self.net.blobs.keys()):
            if layer in layers:
                _layers[layer] = layers[layer]

        for blob in self.net.blobs:
            self.diff[blob] = 0

        input_arr = self._preprocess(np.float32(input_img))
        self.total_px = 0
        self.progress_bar = None
        try:
            detail = self._octave_detail(input_arr, layers=_layers, progress=progress, **kwargs)
        finally:
            if self.progress_bar:
                self.progress_bar.close()
        out = self._deprocess(detail + input_arr)
        if return_ndarray:
            return out
        return to_image(out)
