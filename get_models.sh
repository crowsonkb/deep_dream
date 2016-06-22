#!/bin/bash

BVLC='http://dl.caffe.berkeleyvision.org/bvlc_googlenet.caffemodel'
PLACES205='http://places.csail.mit.edu/model/googlenet_places205.tar.gz'
PLACES365='http://places2.csail.mit.edu/models_places365/googlenet_places365.caffemodel'

( cd bvlc_googlenet && curl -O "$BVLC" )
( cd googlenet_places365 && curl -O "$PLACES365" )
( cd googlenet_places205 &&
  curl "$PLACES205" | tar xzv &&
  cp googlenet_places205/*.caffemodel . &&
  rm -rf googlenet_places205 )
