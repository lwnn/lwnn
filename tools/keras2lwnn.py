import keras2onnx
from onnx2lwnn import *
import re
__all__ = ['keras2lwnn']

def keras2lwnn(model, name, feeds=None):
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx_feeds = {}
    if(feeds != None):
        for inp, v in feeds.items():
            onnx_feeds[inp.name] = v
    onnx2lwnn(onnx_model, name, onnx_feeds)

