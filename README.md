deep_dream
==========

An implementation of the Deep Dream image processing algorithm which is able to process arbitrarily large images.

Requirements:
- Python 3
- [Caffe](http://caffe.berkeleyvision.org), compiled to use Python 3
- [PyPI](https://pypi.python.org/pypi) packages [Pillow](http://pillow.readthedocs.io/en/stable/), [tqdm](https://pypi.python.org/pypi/tqdm)

This implementation of Deep Dream is able to divide the gradient ascent step into tiles if a too-large image is being processed. By default, any image larger than 512x512 will be divided into tiles no larger than 512x512. The tile seams are obscured by applying a random shift on each gradient ascent step (this also greatly improves the image quality by summing over the translation dependence inherent to the neural network architecture).

Locations of pre-trained `.caffemodel` files (place them in their corresponding subdirectories to use them):

- [bvlc_googlenet](http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel)
- [googlenet_places205](http://places.csail.mit.edu/model/googlenet_places205.tar.gz): untar and rename the `.caffemodel` to `googlenet_places205_train_iter_2400000.caffemodel`.
- [googlenet_places365](http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel)
