deep_dream
==========

An implementation of the Deep Dream image processing algorithm which is able to process large (wallpaper-sized) images despite GPU or main memory limits.

Example
-------

![Example input](example_in.jpg)

100 iterations, step size 0.5:

![Example output 1](example_med.jpg)

100 iterations, step size 1.0:

![Example output 2](example_out.jpg)

Requirements
------------

- Python 3.5.
- [Caffe](http://caffe.berkeleyvision.org), compiled to use Python 3.5.
- [PyPI](https://pypi.python.org/pypi) packages [Pillow](http://pillow.readthedocs.io/en/stable/) and [tqdm](https://pypi.python.org/pypi/tqdm) (and Caffe dependencies such as numpy and scikit-image; see its requirements.txt).
- Pre-trained Caffe models (see Models section).

This implementation of Deep Dream is able to divide the gradient ascent step into tiles if a too-large image is being processed. By default, any image larger than 512x512 will be divided into tiles no larger than 512x512. The tile seams are obscured by applying a random shift on each gradient ascent step (this also greatly improves the image quality by summing over the translation dependence inherent to the neural network architecture).

Models
------

Locations of pre-trained `.caffemodel` files (place them in their corresponding subdirectories to use them):

- [bvlc_googlenet](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel): tends toward visualizing abstract patterns, dogs, insects, and amorphous creatures.
- [googlenet_places205](http://places.csail.mit.edu/model/googlenet_places205.tar.gz): tends toward visualizing buildings and landscapes. Untar and rename the `.caffemodel` to `googlenet_places205_train_iter_2400000.caffemodel`.
- [googlenet_places365](http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel): newer than the places205-trained model, more aesthetically pleasing output, tends toward visualizing buildings and landscapes.
