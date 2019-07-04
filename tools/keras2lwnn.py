import keras2onnx
from onnx2lwnn import *

__all__ = ['keras2lwnn']

def keras2lwnn(model, name):
    onnx_model = keras2onnx.convert_keras(model, model.name)
    onnx2lwnn(onnx_model, name)
