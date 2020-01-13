# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

import onnx
import numpy as np

__all__ = ['lwnn2onnx', 'Lwnn2Onnx']

class Lwnn2Onnx():
    def __init__(self, lwnn_model):
        self.lwnn_model = lwnn_model
        self.TRANSLATOR = {
            'Input': self.to_LayerInput,
            'Conv': self.to_LayerConv,
            'Output': self.to_LayerOutput
            }

    def to_LayerInput(self, layer):
        x = onnx.helper.make_tensor_value_info(
            layer['name'], 
            onnx.TensorProto.FLOAT,
            layer['shape'])
        self._inputs.append(x)

    def to_LayerConv(self, layer):
        name = layer['name']
        wname = name + '_weights'
        bname = name + '_bias'
        W = onnx.numpy_helper.from_array(layer['weights'], wname)
        self._initializer.append(W)
        if('bias' in layer):
            B = onnx.numpy_helper.from_array(layer['bias'], bname)
            self._initializer.append(B)
        x = onnx.helper.make_node(
            'Conv',
            inputs = layer['inputs'] + [wname] + [bname] if 'bias' in layer else [],
            outputs = [name],
            shape=layer['shape'])
        self._nodes.append(x)

    def to_LayerOutput(self, layer):
        x = onnx.helper.make_tensor_value_info(
            layer['inputs'][0], 
            onnx.TensorProto.FLOAT,
            layer['shape'])
        self._outputs.append(x)

    def convert(self):
        self._initializer = []
        self._nodes = []
        self._inputs = []
        self._outputs = []
        for ly in self.lwnn_model:
            self.TRANSLATOR[ly['op']](ly)
        graph = onnx.helper.make_graph(
            nodes = self._nodes,
            name = 'lwnn',
            inputs = self._inputs,
            outputs = self._outputs,
            initializer = self._initializer
            )
        model = onnx.helper.make_model(graph, producer_name='lwnn-nhwc')
        return model

    def save(self, onnx_model, p):
        with open(p,'wb') as f:
            f.write(onnx_model.SerializeToString())

def lwnn2onnx(model, p):
    model = Lwnn2Onnx(model)
    onnx_model = model.convert()
    model.save(onnx_model, p)