#!/bin/bash

BVLC='http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
PLACES205='https://s3-us-west-2.amazonaws.com/crowsonkb-deep-dream/googlelet_places205_train_iter_2400000.caffemodel'
PLACES365='http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel'
RESNET50='http://www.deepdetect.com/models/resnet/ResNet-50-model.caffemodel'

( cd bvlc_googlenet && curl -O "$BVLC" )
( cd googlenet_places205 && curl -O "$PLACES205" )
( cd googlenet_places365 && curl -O "$PLACES365" )
( cd resnet && curl -O "$RESNET50" )
