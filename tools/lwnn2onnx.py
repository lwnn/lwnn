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
            'ConvTranspose':  self.to_LayerConv,
            'Relu': self.to_LayerCommon,
            'MaxPool': self.to_LayerCommon,
            'AveragePool': self.to_LayerCommon,
            'Reshape': self.to_LayerCommon,
            'Dense': self.to_LayerDense,
            'Concat': self.to_LayerCommon,
            'Pad': self.to_LayerCommon,
            'Softmax': self.to_LayerCommon,
            'Add': self.to_LayerCommon,
            'Upsample': self.to_LayerCommon,
            'BatchNormalization': self.to_LayerBatchNormalization,
            'Output': self.to_LayerOutput
            }

    def to_LayerCommon(self, layer, **kwargs):
        name = layer['name']
        attr = {}
        for k,v in layer.items():
            if(k not in ['name', 'outputs', 'op']):
                attr[k] = v
        for k,v in kwargs.items(): # handle default attr
            if(k not in layer):
                attr[k] = v
        x = onnx.helper.make_node(
            layer['op'],
            name = name,
            outputs=[name],
            **attr)
        self._nodes.append(x)

    def to_LayerInput(self, layer):
        x = onnx.helper.make_tensor_value_info(
            layer['name'], 
            onnx.TensorProto.FLOAT,
            layer['shape'])
        self._inputs.append(x)

    def get_attr(self, layer, attr_name, attr_default=None):
        if(attr_name in layer):
            return layer[attr_name]
        elif(attr_default is not None):
            return attr_default
        else:
            raise Exception('attr %s not found for layer %s'%(attr_name, layer))

    def get_attrs(self, layer, exclude, **kwargs):
        attr = {}
        for k,v in layer.items():
            if(k not in exclude):
                attr[k] = v
        for k,v in kwargs.items(): # handle default attr
            if(k not in layer):
                attr[k] = v
        return attr

    def to_LayerConv(self, layer):
        name = layer['name']
        op = layer['op']
        shape=layer['shape']
        wname = name + '_weights'
        bname = name + '_bias'
        W = layer['weights']
        group = layer['group']
        C = shape[1]
        if(op == 'Conv'): # reorder weights to [M x C/group x kH x kW]
            if(group == C): # DwConv [C/group x kH x kW x M]
                W = W.transpose(3,0,1,2)
            else: # Conv [M x kH x kW x C/group]
                W = W.transpose(0,3,1,2)
        else: # ConvTranspose [M/group x kH x kW x C] -> (C x M/group x kH x kW)
            W = np.rot90(W, k=2, axes=(1, 2))
            W = W.transpose(3,0,1,2)
        attr = self.get_attrs(layer,
            ['name', 'outputs', 'op', 'inputs', 'activation', 'alpha',
             'weights', 'bias'],
            strides=[1, 1],
            dilations=[1, 1],
            kernel_shape = W.shape[-2:],
            auto_pad='NOTSET')
        W = onnx.numpy_helper.from_array(W, wname)
        self._initializer.append(W)
        if('bias' in layer):
            B = onnx.numpy_helper.from_array(layer['bias'], bname)
            self._initializer.append(B)
        conv_name = name
        if(('activation' in layer) and (layer['activation'] != 'linear')):
            conv_name = name + '_o'
        x = onnx.helper.make_node(
            op,
            name = conv_name,
            inputs = layer['inputs'] + [wname] + [bname] if 'bias' in layer else [],
            outputs = [conv_name],
            **attr)
        self._nodes.append(x)
        if(('activation' in layer) and (layer['activation'] != 'linear')):
            activation = layer['activation']
            if(activation == 'Relu'):
                x = onnx.helper.make_node(
                        'Relu',
                        name = name,
                        inputs = [conv_name],
                        outputs = [name],
                        shape = shape)
            elif(activation == 'leaky'):
                x = onnx.helper.make_node(
                        'LeakyRelu',
                        name = name,
                        inputs = [conv_name],
                        outputs = [name],
                        alpha=self.get_attr(layer, 'alpha', 0.1),
                        shape = shape)
            else:
                raise NotImplementedError('activation %s is not supported'%(activation))
            self._nodes.append(x)

    def to_LayerBatchNormalization(self, layer):
        name = layer['name']
        inputs = layer['inputs']
        for i in ['scale', 'bias', 'mean', 'var']:
            self._initializer.append(onnx.numpy_helper.from_array(layer[i], '%s_%s'%(name, i)))
            inputs.append('%s_%s'%(name, i))
        x = onnx.helper.make_node(
            layer['op'],
            name = name,
            inputs=inputs,
            outputs=[name],
            epsilon=self.get_attr(layer, 'epsilon', 1e-05),
            momentum =self.get_attr(layer, 'momentum ', 0.9),
            shape = layer['shape'])
        self._nodes.append(x)

    def to_LayerDense(self, layer):
        name = layer['name']
        wname = name + '_weights'
        bname = name + '_bias'
        W = layer['weights']
        B = layer['bias']
        W = onnx.numpy_helper.from_array(W, wname)
        B = onnx.numpy_helper.from_array(B, bname)
        self._initializer.extend([W,B])
        x = onnx.helper.make_node(
            'MatMul',
            name = name+'_o',
            inputs=layer['inputs']+[wname],
            outputs=[name+'_o'],
            shape = layer['shape'])
        self._nodes.append(x)
        x = onnx.helper.make_node(
            'Add',
            name = name,
            inputs=[name+'_o', bname],
            outputs=[name],
            shape = layer['shape'])
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