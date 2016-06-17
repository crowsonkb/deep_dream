deep_dream
==========

An implementation of the Deep Dream image processing algorithm which is able to process arbitrarily large images.

Requirements:
- Python 3
- [Caffe](http://caffe.berkeleyvision.org), compiled to use Python 3
- PyPI packages Pillow, tqdm

This implementation of Deep Dream is able to divide the gradient ascent step into tiles if a too-large image is being processed. By default, any image larger than 512x512 will be divided into tiles no larger than 512x512. The tile seams are obscured by applying a random shift on each gradient ascent step (this also greatly improves the image quality by summing over the translation dependence inherent to the neural network architecture).
