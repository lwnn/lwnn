import keras2onnx
from onnx2lwnn import *
import os
__all__ = ['keras2lwnn']

def keras2lwnn(model, name, feeds=None):
    onnx_model = keras2onnx.convert_keras(model, model.name,
                        channel_first_inputs=[model.input])
    onnx_feeds = {}
    if(feeds != None):
        for inp, v in feeds.items():
            onnx_feeds[inp.name] = v
    onnx2lwnn(onnx_model, name, onnx_feeds)
    if('1' == os.getenv('LWNN_GTEST')):
        os.makedirs('models/%s'%(name), exist_ok=True)
        model.save('models/%s/%s.h5'%(name,name))
