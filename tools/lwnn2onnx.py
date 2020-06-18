# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

import onnx
import onnx.helper
import onnx.numpy_helper
import numpy as np

__all__ = ['lwnn2onnx', 'Lwnn2Onnx']

class Lwnn2Onnx():
    def __init__(self, lwnn_model):
        self.lwnn_model = lwnn_model
        self.TRANSLATOR = {
            'Input': self.to_LayerInput,
            'Conv': self.to_LayerConv,
            'ConvTranspose':  self.to_LayerConv,
            'Dense': self.to_LayerDense,
            'BatchNormalization': self.to_LayerBatchNormalization,
            'Scale': self.to_LayerScale,
            'Normalize': self.to_LayerNormalize,
            'Constant': self.to_LayerConst,
            'LSTM': self.to_LayerLSTM,
            'Output': self.to_LayerOutput
            }

    def to_LayerCommon(self, layer, initializer=[], **kwargs):
        name = layer['name']
        attr = {}
        inputs = list(layer['inputs']) if 'inputs' in layer else []
        for k,v in layer.items():
            if((type(v) == np.ndarray) and (k not in initializer)):
                initializer.append(k)
        for i in initializer:
            if(i not in layer):
                raise Exception('layer %s has no member %s'%(layer, i))
            self._initializer.append(onnx.numpy_helper.from_array(layer[i], '%s_%s'%(name, i)))
            inputs.append('%s_%s'%(name, i))
        for k,v in layer.items():
            if(k not in ['name', 'outputs', 'inputs', 'op', 'shape']+initializer):
                if(type(v) in [list, tuple]):
                    types = [type(x) for x in v]
                    if((float in types) and (int in types)):
                        v = [float(x) for x in v]
                attr[k] = v
        for k,v in kwargs.items(): # handle default attr
            if(k not in layer):
                attr[k] = v
        try:
            x = onnx.helper.make_node(
                layer['op'],
                name = name,
                outputs=[name],
                inputs=inputs,
                **attr)
        except Exception as e:
            raise Exception('%s: %s'%(layer, e))
        self._nodes.append(x)
        if('shape' in layer):
            vinfo = onnx.helper.make_tensor_value_info(
                name,
                onnx.TensorProto.FLOAT, 
                layer['shape'])
            self._value_info.append(vinfo)

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
        if('activation' in layer):
            activation = layer['activation'].lower()
        else:
            activation = 'linear'
        if(activation != 'linear'):
            conv_name = name + '_o'
        x = onnx.helper.make_node(
            op,
            name = conv_name,
            inputs = layer['inputs'] + [wname] + [bname] if 'bias' in layer else [],
            outputs = [conv_name],
            **attr)
        self._nodes.append(x)
        if(activation != 'linear'):
            if(activation == 'relu'):
                x = onnx.helper.make_node(
                        'Relu',
                        name = name,
                        inputs = [conv_name],
                        outputs = [name])
            elif(activation == 'leaky'):
                x = onnx.helper.make_node(
                        'LeakyRelu',
                        name = name,
                        inputs = [conv_name],
                        outputs = [name],
                        alpha=self.get_attr(layer, 'alpha', 0.1))
            elif(activation == 'sigmoid'):
                x = onnx.helper.make_node(
                        'Sigmoid',
                        name = name,
                        inputs = [conv_name],
                        outputs = [name])
            else:
                raise NotImplementedError('activation %s is not supported'%(activation))
            self._nodes.append(x)
        vinfo = onnx.helper.make_tensor_value_info(
                name,
                onnx.TensorProto.FLOAT, 
                shape)
        self._value_info.append(vinfo)

    def to_LayerBatchNormalization(self, layer):
        self.to_LayerCommon(layer, ['scale', 'bias', 'mean', 'var'],
            epsilon=self.get_attr(layer, 'epsilon', 1e-05),
            momentum =self.get_attr(layer, 'momentum ', 0.9))

    def to_LayerScale(self, layer):
        self.to_LayerCommon(layer, ['weights', 'bias'])

    def to_LayerNormalize(self, layer):
        self.to_LayerCommon(layer, ['scale'])

    def to_LayerConst(self, layer):
        name = layer['name']
        value = onnx.numpy_helper.from_array(layer.const if 'const' in layer else layer.value)
        x = onnx.helper.make_node(
            'Constant',
            name=name,
            inputs=[],
            outputs=[name],
            value=value)
        self._nodes.append(x)
        if('shape' not in layer): return
        vinfo = onnx.helper.make_tensor_value_info(
                name,
                value.data_type, 
                layer['shape'])
        self._value_info.append(vinfo)

    def to_LayerDense(self, layer):
        self.to_LayerCommon(layer, ['weights', 'bias'])

    def to_LayerLSTM(self, layer):
        self.to_LayerCommon(layer, ['W', 'R', 'B'])

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
        self._value_info = []
        for ly in self.lwnn_model:
            op = ly['op']
            if(op in self.TRANSLATOR):
                translator = self.TRANSLATOR[op]
            else:
                translator = self.to_LayerCommon
            if(translator == self.to_LayerCommon):
                self.to_LayerCommon(ly, []) # fix issue that sometimes initializer != []
            else:
                translator(ly)
        graph = onnx.helper.make_graph(
            nodes = self._nodes,
            name = 'lwnn',
            inputs = self._inputs,
            outputs = self._outputs,
            value_info = self._value_info,
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