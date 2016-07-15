"""Deep Dreaming using Caffe and Google's Inception convolutional neural network."""

# pylint: disable=invalid-name, wrong-import-position

from collections import namedtuple, OrderedDict
import multiprocessing as mp
import os
from pathlib import Path
import queue
import re

os.environ['GLOG_minloglevel'] = '1'
import caffe
import numpy as np
from PIL import Image
from scipy import ndimage
from skimage.restoration import denoise_tv_bregman
from tqdm import tqdm

from .tile_worker import TileRequest, TileWorker

CTX = mp.get_context('spawn')
EPS = np.finfo(np.float32).eps

# """A smoothing kernel - Sobel weight matrix and truncated Gaussian (std 1.2).
#    (http://www.hlevkin.com/articles/SobelScharrGradients5x5.pdf)"""
# KERNEL = np.sqrt(np.float32([[[1, 2, 1], [2, 4, 2], [1, 2, 1]]]))
# KERNEL /= KERNEL.sum()

CNNData = namedtuple('CNNData', 'deploy model mean categories')
CNNData.__new__.__defaults__ = (None,)  # Make categories optional.

_BASE_DIR = Path(__file__).parent.parent
GOOGLENET_BVLC = CNNData(
    _BASE_DIR/'bvlc_googlenet/deploy.prototxt',
    _BASE_DIR/'bvlc_googlenet/bvlc_googlenet.caffemodel',
    (104, 117, 123),
    categories=_BASE_DIR/'bvlc_googlenet/categories.txt')
GOOGLENET_PLACES205 = CNNData(
    _BASE_DIR/'googlenet_places205/deploy_places205.prototxt',
    _BASE_DIR/'googlenet_places205/googlelet_places205_train_iter_2400000.caffemodel',
    (104.051, 112.514, 116.676),  # TODO: find the actual Places205 mean
    categories=_BASE_DIR/'googlenet_places205/categories.txt')
GOOGLENET_PLACES365 = CNNData(
    _BASE_DIR/'googlenet_places365/deploy_googlenet_places365.prototxt',
    _BASE_DIR/'googlenet_places365/googlenet_places365.caffemodel',
    (104.051, 112.514, 116.676),
    categories=_BASE_DIR/'googlenet_places365/categories_places365.txt')


def call_normalized(fn, arr, *args, **kwargs):
    normed = arr.copy()
    offset = normed.min()
    normed -= offset
    scale = normed.max()
    normed /= scale
    ret = fn(normed, *args, **kwargs)
    if isinstance(ret, np.ndarray):
        return ret * scale + offset
    return ret


def save_as_hdr(arr, filename, gamma=2.2, allow_negative=True):
    """Saves a float32 ndarray to a high dynamic range (OpenEXR or float32 TIFF) file.

    Args:
        arr (ndarray): The input array.
        filename (str | Path): The output filename.
        gamma (Optional[float]): The encoding gamma of arr.
        allow_negative (Optional[bool]): Clip negative values to zero if false."""
    arr = arr.astype(np.float32)/255
    if not allow_negative:
        arr[arr < 0] = 0
    if gamma != 1:
        arr = np.sign(arr)*np.abs(arr)**gamma
    filename = str(filename)
    extension = filename.rpartition('.')[2].lower()
    if extension == 'exr':
        import OpenEXR
        exr = OpenEXR.OutputFile(filename, OpenEXR.Header(arr.shape[1], arr.shape[0]))
        exr.writePixels({'R': arr[..., 0].tobytes(),
                         'G': arr[..., 1].tobytes(),
                         'B': arr[..., 2].tobytes()})
        exr.close()
    elif extension == 'tif' or extension == 'tiff':
        import tifffile
        tiff = tifffile.TiffWriter(filename)
        tiff.save(arr, photometric='rgb')
        tiff.close()
    else:
        raise Exception('Unknown HDR file format.')


def to_image(arr):
    """Clips the values in a float32 ndarray to 0-255 and converts it to a PIL image.

    Args:
        arr (ndarray): The input array."""
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
        return 'bad shape %s' % self.args


class CaffeStateError(Exception):
    """Raised by CNN when the worker processes have died or malfunctioned, or Caffe is otherwise
       in a bad state. This error is only dealable with by creating a new CNN instance."""
    def __str__(self):
        return 'Bad Caffe state: %s' % self.args


class _LayerIndexer:
    def __init__(self, net, attr):
        self.net, self.attr = net, attr

    def __getitem__(self, key):
        return getattr(self.net.blobs[key], self.attr)[0]

    def __setitem__(self, key, value):
        getattr(self.net.blobs[key], self.attr)[0] = value


class CNN:
    """Represents an instance of a Caffe convolutional neural network."""

    def __init__(self, cnndata, cpu_workers=0, gpus=[]):
        """Initializes a CNN.

        Example:
            CNN(GOOGLENET_PLACES365, cpu_workers=0, gpus=[0])

        Args:
            cpu_workers (Optional[int]): The number of CPU workers to start. The default is 1 if
                no other compute devices are specified.
            gpus (Optional[list[int]]): The GPU device numbers to start GPU workers on.
        """
        caffe.set_mode_cpu()
        self.net = caffe.Classifier(str(cnndata.deploy), str(cnndata.model),
                                    mean=np.float32(cnndata.mean), channel_swap=(2, 1, 0))
        self.data = _LayerIndexer(self.net, 'data')
        self.diff = _LayerIndexer(self.net, 'diff')
        self.categories = [str(i) for i in range(self.data['prob'].size)]
        if cnndata.categories is not None:
            self.categories = open(str(cnndata.categories)).read().splitlines()
        self.img = np.zeros_like(self.data['data'])
        self.total_px = 0
        self.progress_bar = None
        self.req_q = CTX.JoinableQueue()
        self.resp_q = CTX.Queue()
        self.workers = []
        self.is_healthy = True
        if not cpu_workers and not gpus:
            cpu_workers = 1
        for _ in range(cpu_workers):
            self.workers.append(TileWorker(self.req_q, self.resp_q, cnndata, None))
        for gpu in gpus:
            self.workers.append(TileWorker(self.req_q, self.resp_q, cnndata, gpu))

    def __del__(self):
        self.is_healthy = False
        for worker in self.workers:
            worker.__del__()

    def ensure_healthy(self):
        """Ensures that the worker subprocesses are healthy. If one has terminated, it will
           terminate the others, set self.is_healthy to False, and raise a CaffeStateError."""
        if not self.is_healthy:
            raise CaffeStateError('The worker processes were terminated. Please make a new CNN instance.')
        for worker in self.workers:
            if worker.proc.exitcode:
                self.__del__()
                raise CaffeStateError('Worker process malfunction detected; terminating others.')
        return True

    def _preprocess(self, img):
        return np.rollaxis(np.float32(img), 2)[::-1] - self.net.transformer.mean['data']

    def _deprocess(self, img):
        return np.dstack((img + self.net.transformer.mean['data'])[::-1])

    def get_features(self, input_img, layers=None, max_tile_size=512):
        """Retrieve feature maps from the classification (forward) phase of operation.

        Example:
            cnn.get_features(img, ['prob'])['prob'] classifies 'img' and returns the predicted
            probability distribution over the network's categories.

        Returns:
            A dict which maps each layer in layers to a retrieved feature map.
        """
        input_arr = self._preprocess(np.float32(input_img))
        h = min(max_tile_size, input_arr.shape[1])
        w = min(max_tile_size, input_arr.shape[2])
        if max(*input_arr.shape[1:]) > max_tile_size:
            input_arr = _resize(input_arr, (h, w))
        self.net.blobs['data'].reshape(1, 3, h, w)
        self.data['data'] = input_arr
        end = self.layers()[-1]
        self.net.forward(end=end)
        if not layers:
            layers = self.layers()
        return {layer: self.data[layer].copy() for layer in layers}

    def _grad_tiled(self, layers, progress=False, max_tile_size=512, **kwargs):
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
                sy, sx = h//ny*y, w//nx*x

                data = self.img[:, sy:sy+th, sx:sx+tw]
                self.ensure_healthy()
                self.req_q.put(TileRequest((sy, sx), data, layers, kwargs))

        for _ in range(ny*nx):
            while True:
                try:
                    self.ensure_healthy()
                    resp, grad = self.resp_q.get(True, 1)
                    break
                except queue.Empty:
                    continue
            sy, sx = resp
            g[:, sy:sy+grad.shape[1], sx:sx+grad.shape[2]] = grad
            if progress:
                self.progress_bar.update(np.prod(grad.shape[-2:]))

        return g

    def _step(self, n=1, step_size=1.5, jitter=32, seed=0, smoothing=0, tv_weight=None, **kwargs):
        np.random.seed(self.img.size + seed)
        for _ in range(n):
            x, y = np.random.randint(-jitter, jitter+1, 2)
            self.img = np.roll(np.roll(self.img, x, 2), y, 1)
            g = self._grad_tiled(**kwargs)
            g /= np.mean(np.abs(g)) + EPS
            self.img += step_size * g
            self.img = np.roll(np.roll(self.img, -x, 2), -y, 1)
            if smoothing:
                smoothed = ndimage.gaussian_filter(self.img, (0, 1, 1), mode='nearest', truncate=2)
                self.img = self.img*(1-smoothing) + smoothed*smoothing
        if tv_weight is not None:
            self.img = call_normalized(denoise_tv_bregman, self.img.T, tv_weight).T

    def _octave_detail(self, base, scales=4, min_size=32, per_octave=2, fn=None, **kwargs):
        if 'n' not in kwargs:
            kwargs['n'] = 10
        n = kwargs['n']
        fnargs = {}
        if fn:
            fnargs.update(fn(base.shape[-2:]))
            if 'n' in fnargs:
                n = fnargs['n']
        if min(base.shape[1:]) < 32:
            raise ShapeError(base.shape)

        factor = 2**(1/per_octave)
        detail = np.zeros_like(base, dtype=np.float32)
        self.total_px += base.shape[1] * base.shape[2] * n
        if scales != 1:
            hf, wf = np.int32(np.round(np.array(base.shape)[-2:]/factor))
            if min(hf, wf) >= min_size:
                smaller_base = _resize(base, (hf, wf))
                smaller_detail = self._octave_detail(
                    smaller_base, scales-1, min_size, per_octave, fn, **kwargs)
                detail = _resize(smaller_detail, base.shape[-2:])
        self.img = base + detail
        kwargs.update(fnargs)
        self._step(**kwargs)
        return self.img - base

    def layers(self, pattern='.*'):
        """Returns a list of layer names matching a regular expression."""
        layers = []
        for i, layer in enumerate(self.net.blobs.keys()):
            if i == 0 or layer.partition('_split_')[1]:
                continue
            if re.fullmatch(pattern, layer):
                layers.append(layer)
        return layers

    def classify(self, input_img, n=1, **kwargs):
        """Classifies the input image and returns the n most probable categories.

        Args:
            input_img: The image to process (PIL images or Numpy arrays are accepted).
            n: The n most probable categories to return.
            max_tile_size: Does not allow the image dimension to exceed this.

        Returns:
            A list containing the n most probable categories."""
        prob = self.get_features(input_img, ['prob'], **kwargs)['prob']
        indices = prob.argsort()[::-1][:n]
        return [(prob[i], self.categories[i]) for i in indices]

    def prepare_layer_list(self, layers):
        if isinstance(layers, str):
            layers = [layers]
        if isinstance(layers, list):
            layers = {layer: 1 for layer in layers}
        _layers = OrderedDict()
        for layer in reversed(self.net.blobs.keys()):
            if layer in layers:
                _layers[layer] = layers[layer]
        return _layers

    def prepare_guide_weights(self, guide_img, layers=None, max_guide_size=512):
        if not layers:
            layers = self.layers()
        if isinstance(layers, str):
            layers = [layers]
        guide_features = self.get_features(guide_img, layers, max_tile_size=max_guide_size)
        weights = {}
        for layer in layers:
            if guide_features[layer].ndim != 3:
                continue
            v = guide_features[layer].sum(1).sum(1)[:, None, None]
            weights[layer] = v/np.abs(v).sum()**2
        return self.prepare_layer_list(weights)

    def subset_layers(self, layers, new_layers):
        _layers = OrderedDict()
        for layer in new_layers:
            _layers[layer] = layers[layer]
        return _layers

    def dream(self, input_img, layers, progress=True, **kwargs):
        """Runs the Deep Dream multiscale gradient ascent algorithm on the input image.

        Args:
            input_img: The image to process (PIL images or Numpy arrays are accepted)
            layers (dict): The layer/feature weights to use in the objective function for gradient
                ascent.
            progress (Optional[bool]): Display a progress bar while computing.
            scales (Optional[int]): The number of scales to process.
            min_size (Optional[int]): Don't permit the small edge of the image to go below this.
            per_octave (Optional[int]): Determines the difference between each scale; for instance,
                the default of 2 means that a 1000x1000 input image will get processed as 707x707
                and 500x500.
            n (Optional[int]): The number of gradient ascent steps per scale. Defaults to 10.
            step_size (Optional[float]): The strength of each individual gradient ascent step.
                Specifically, each step will change the image's pixel values by a median of
                step_size.
            smoothing (Optional[float]): The factor to smooth the image by after each gradient
                ascent step. Try 0.02-0.1.
            tv_weight (Optional[float]): The denoising weight for total variation regularization,
                performed at the end of each scale. Higher values smooth the image less.
                Try 25-100.
            max_tile_size (Optional[int]): Defaults to 512, suitable for a GPU with 2 GB RAM.
                Higher values perform better; if Caffe runs out of GPU memory and crashes then it
                should be lowered.

        Returns:
            The unclipped processed image as a float32 ndarray which has a valid range of 0-255 but
            which may contain components that are less than 0 or greater than 255.
            deep_dream.to_image() can be used to convert the ndarray to a PIL image.
        """
        self.ensure_healthy()
        _layers = self.prepare_layer_list(layers)
        input_arr = self._preprocess(np.float32(input_img))
        self.total_px = 0
        self.progress_bar = None
        try:
            detail = self._octave_detail(input_arr, layers=_layers, progress=progress, **kwargs)
        except KeyboardInterrupt:
            self.__del__()
            raise CaffeStateError('Worker processes left in inconsistent states. Terminating them.')
        finally:
            if self.progress_bar:
                self.progress_bar.close()
        return self._deprocess(detail + input_arr)

    def dream_guided(self, input_img, guide_img, layers, max_guide_size=512, **kwargs):
        """Performs guided gradient ascent on input_img, weighted by the feature map channel sums
        of guide_img. This algorithm works best using a relatively large number of layers, such as
        (for googlenet) anything matching the regular expression 'inception_../output'. The
        relative weights of the layers are determined automatically."""
        self.ensure_healthy()
        weights = self.prepare_guide_weights(guide_img, layers, max_guide_size)
        return self.dream(input_img, weights, auto_weight=False, **kwargs)
