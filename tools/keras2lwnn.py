import keras2onnx
from onnx2lwnn import *

__all__ = ['keras2lwnn']

def to_onnx_layer_name(name):
    v = name.split(':')
    n = v[0]
    i = eval(v[1])+1
    return '%s_%02d'%(n,i)

def keras2lwnn(model, name, feeds=None):
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx_feeds = {}
    if(feeds != None):
        for inp, v in feeds.items():
            onnx_feeds[to_onnx_layer_name(inp.name)] = v
    onnx2lwnn(onnx_model, name, onnx_feeds)

