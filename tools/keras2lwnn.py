import keras2onnx
from onnx2lwnn import *
import re
__all__ = ['keras2lwnn']

reLoadName = re.compile(r'([^\s]+)_(\d+)_(\d+)') 
def to_onnx_layer_name(name):
    v = name.split(':')
    n = v[0]
    i = eval(v[1])+1
    if(reLoadName.search(n)):
        grp = reLoadName.search(n).groups()
        n = '_'.join(grp[:-1])
    if(keras2onnx.__version__ == '1.5.0'):
        return '%s_%02d'%(n,i)
    else:
        return n

def keras2lwnn(model, name, feeds=None):
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx_feeds = {}
    if(feeds != None):
        for inp, v in feeds.items():
            onnx_feeds[to_onnx_layer_name(inp.name)] = v
    onnx2lwnn(onnx_model, name, onnx_feeds)

