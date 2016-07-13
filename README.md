deep_dream
==========

An implementation of the Deep Dream image processing algorithm which is able to process large (wallpaper-sized) images despite GPU or main memory limits.

Requirements
------------

- Python 3.5.
- [Caffe](http://caffe.berkeleyvision.org), built against Python 3.5. I would encourage you to use Caffe's nVidia GPU support if possible: it runs several times faster on even a laptop GPU (GeForce GT 750M) than on the CPU.
  - OS X:
    - Caffe + Python 3.5 `Makefile.config` settings, with python3 installed through [homebrew](http://brew.sh):
    ```make
    PYTHON_DIR := /usr/local/opt/python3/Frameworks/Python.framework/Versions/3.5
    PYTHON_INCLUDE := $(PYTHON_DIR)/include/python3.5m \
            /usr/local/lib/python3.5/site-packages/numpy/core/include
    PYTHON_LIB := $(PYTHON_DIR)/lib
    PYTHON_LIBRARIES := boost_python3 python3.5m
    ```
    - This assumes you installed numpy with pip into the python3.5 system   site-packages directory. If you're in a virtualenv this may change.
    - Leave the `BLAS` setting at `atlas`, unless you want to try MKL (faster). Recent OS X ships with an optimized multithreaded BLAS so there is little reason IMO to use openblas anymore.
- [PyPI](https://pypi.python.org/pypi) packages [Pillow](http://pillow.readthedocs.io/en/stable/) and [tqdm](https://pypi.python.org/pypi/tqdm) (and Caffe dependencies such as numpy and scikit-image; see its requirements.txt).
- [openexrpython](https://github.com/jamesbowman/openexrpython), installed from git master instead of 1.2.0 from PyPI, if you are going to use OpenEXR (high dynamic range) export. (`pip install -U git+https://github.com/jamesbowman/openexrpython`)
- Pre-trained Caffe models (run `get_models.sh`; see [Models](#models) section).

This implementation of Deep Dream is able to divide the gradient ascent step into tiles if a too-large image is being processed. By default, any image larger than 512x512 will be divided into tiles no larger than 512x512. The tile seams are obscured by applying a random shift on each gradient ascent step (this also greatly improves the image quality by summing over the translation dependence inherent to the neural network architecture).

Example
-------

```
import deep_dream as dd
from PIL import Image

cnn = dd.CNN(dd.GOOGLENET_PLACES365, gpu=0)
img = Image.open('kodim/img0022.jpg').resize((768, 512), Image.LANCZOS)
```

<img src="example_in.jpg" width="384" height="256">

```
out = cnn.dream(img, 'inception_4a/output', scale=12, per_octave=4, n=8, step_size=0.5)
dd.to_image(out).save('example_med.jpg', quality=85)
```

<img src="example_med.jpg" width="384" height="256">

```
out = cnn.dream(img, 'inception_4a/output', scale=12, per_octave=4, n=12, step_size=1.0)
dd.to_image(out).save('example_out.jpg', quality=85)
```

<img src="example_out.jpg" width="384" height="256">

CNN.dream_guided() example
--------------------------

Input:  
<img src="example2_in.jpg" width="512" height="341">

Guide:  
<img src="example2_guide.jpg" width="512" height="341">

Combined output:  
<img src="example2_out.jpg" width="512" height="341">

Gradient ascent was performed using layers `inception_(3a-b, 4a-e, 5a-b)/output`. This is a reasonable set of layers for `dream_guided()` to work well. Note that the input and the guide do not have to be the same size; the output will be the same size as the input.

Models
------

Locations of pre-trained `.caffemodel` files (place them in their corresponding subdirectories to use them):

- [bvlc_googlenet](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel): tends toward visualizing abstract patterns, dogs, insects, and amorphous creatures.
- [googlenet_places205](http://places.csail.mit.edu/model/googlenet_places205.tar.gz): tends toward visualizing buildings and landscapes.
- [googlenet_places365](http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel): newer than the places205-trained model, often more aesthetically pleasing output, tends toward visualizing buildings and landscapes.
