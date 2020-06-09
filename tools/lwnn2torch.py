# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import os
import pickle
import numpy as np
import glob
import liblwnn as lwnn
from lwnn import *
from onnx2lwnn import OnnxConverter
# ref https://pytorch.org/docs/stable/_modules/torch/nn/quantized/functional.html
import torch
from verifyoutput import *

__all__ = ['lwnn2torch', 'Lwnn2Torch']

def LI(a):
    return np.asarray(a, dtype=np.int32)

class Lwnn2Torch(torch.nn.Module):
    def __init__(self, p, **kwargs):
        super(Lwnn2Torch, self).__init__()
        self.kwargs = kwargs
        if(type(p) == str):
            self.lwnn_model = self.load(p)
        else:
            self.lwnn_model = p
        self.convert()

    def load(self, p):
        converter = OnnxConverter(p, lwnn=True)
        return converter.model

    def convert(self):
        for layer in self.lwnn_model:
            cvtfunc = getattr(self, 'to_Layer%s'%(layer.op))
            cvtfunc(layer)

    @property
    def inputs(self):
        inps = {}
        shapes = self.kwargs['shapes'] if 'shapes' in self.kwargs else {}
        for ly in self.lwnn_model:
            if(ly['op'] == 'Input'):
                if(ly.name in shapes):
                    inps[ly.name] = shapes[ly.name]
                else:
                    inps[ly.name] = ly.shape
        return inps

    def get_layers(self, names):
        layers = []
        for layer in self.lwnn_model:
            if(layer['name'] in names):
                layers.append(layer)
        return layers

    def __calc_SZ_qint8(self, min, max):
        if((min==0.0) and (max==0.0)):
            scale = 1
            middle = 0
        else:
            middle = (min+max)/2
            min = min - middle
            max = max - middle
            scale = max/(127.0/(2**7))
        S = scale/(2**7)  # always 7 fraction bits for lwnn
        Z = -np.round(middle/S).astype(np.int32)
        return S,int(Z)

    def __calc_SZ_quint8(self, min, max):
        if((min==0.0) and (max==0.0)):
            Z = 0
            S = 1.0
        else:
            # adjust max/min to make sure zero is integer
            zf = -255.0*min/(max-min)
            zi = np.floor(zf)
            of = zf - zi
            if(of < 10e-6):
                pass
            elif(zi <= 0):
                zi = 0
                min = 0.0
            elif(zi >= 255):
                max = 0.0
            elif(of < 0.5):
                max = -255.0*min/zi+min
            else:
                zi += 1
                min = -zi/(255.0-zi)*max
            Z = zi
            S = (max-min)/255.0
        return S,int(Z)

    def calc_SZ(self, bottom, dtype=torch.quint8):
        # TODO: lwnn is using qint8 for both input/weights, 
        # but for now torch conv2d only support quint8 input and qint8 weights
        min = np.min(bottom)
        max = np.max(bottom)
        if(dtype == torch.qint8):
            S,Z = self.__calc_SZ_qint8(min, max)
        elif(dtype == torch.quint8):
            S,Z = self.__calc_SZ_quint8(min, max)
        else:
            raise NotImplementedError('dtype <%s> is not supported'%(dtype))
        return S,Z

    def quantize_per_tensor(self, bottom, dtype=torch.quint8):
        S,Z = self.calc_SZ(bottom, dtype)
        top = torch.nn.quantized.Quantize(S, Z, dtype)(torch.from_numpy(bottom))
        return top

    def get_attr(self, layer, attr_name, attr_default=None):
        if(attr_name in layer):
            return layer[attr_name]
        elif(attr_default is not None):
            return attr_default
        else:
            raise Exception('attr %s not found for layer %s'%(attr_name, layer))

    def get_bottom(self, layer, index=0):
        return self._outputs[layer.inputs[index]]
    def set_top(self, layer, top, index=0):
        self._outputs[layer.outputs[index]] = top
    def get_top(self, layer, index=0):
        return self._outputs[layer.outputs[index]]

    def to_LayerInput(self, layer):
        pass
    def forward_LayerInput(self, layer):
        pass

    def forward_LayerCommon(self, layer):
        bottom = self.get_bottom(layer)
        op = getattr(self, layer.name)
        top = op(bottom)
        self.set_top(layer, top)

    def to_LayerConv(self, layer, Q=False):
        inp = self.get_layers(layer['inputs'])[0]
        W = layer['weights']
        group = layer.group
        B = layer['bias']
        Cin = int(W.shape[1]/group)
        Cout = W.shape[0]
        strides = self.get_attr(layer, 'strides', [1, 1])
        dilation = self.get_attr(layer, 'dilations', [1, 1])
        if(Q==True):
            bottom = inp['topq'][0]
            if(bottom is None):
                layer['topq'] = [None]
                return
            # using the float output to inference the scale,zero_point
            S,Z = self.calc_SZ(layer['top'][0])
            top = torch.nn.quantized.functional.conv2d(
                     bottom,
                     weight=self.quantize_per_tensor(W, torch.qint8),
                     bias=torch.from_numpy(B),
                     stride=strides,
                     dilation = dilation,
                     padding=layer['pads'][:2],
                     groups=layer['group'],
                     scale = S,
                     zero_point = Z,
                     dtype=torch.quint8)
            layer['topq'] = [top]
        else:
            conv = torch.nn.Conv2d(in_channels=Cin,
                     out_channels=Cout,
                     kernel_size=list(W.shape)[2:],
                     stride=strides,
                     dilation = dilation,
                     padding=layer['pads'][:2],
                     groups=layer['group'],
                     bias=True)
            conv.weight = torch.nn.Parameter(torch.from_numpy(W))
            conv.bias = torch.nn.Parameter(torch.from_numpy(B))
            setattr(self, layer.name, conv)

    def to_QLayerConv(self, layer):
        return self.to_LayerConv(layer, Q=True)

    def to_LayerDense(self, layer):
        W = layer['weights']
        B = layer['bias']
        in_features = W.shape[0]
        out_features = W.shape[1]
        dense = torch.nn.Linear(in_features, out_features)
        dense.weight = torch.nn.Parameter(torch.from_numpy(W.transpose()))
        dense.bias = torch.nn.Parameter(torch.from_numpy(B))
        setattr(self, layer.name, dense)

    def to_LayerRelu(self, layer):
        setattr(self, layer.name, torch.nn.ReLU())

    def to_LayerReshape(self, layer):
        pass
    def forward_LayerReshape(self, layer):
        bottom = self.get_bottom(layer)
        shapes = self.kwargs['shapes'] if 'shapes' in self.kwargs else {}
        if(layer.name in shapes):
            shape = shapes[layer.name]
        else:
            shape = layer.shape
        shape = [-1]+list(shape)[1:]
        top = torch.reshape(bottom, shape)
        self.set_top(layer, top)

    def to_LayerMin(self, layer):
        pass
    def forward_LayerMin(self, layer):
        a = self.get_bottom(layer, 0)
        b = self.get_bottom(layer, 1)
        top = torch.min(a, b)
        self.set_top(layer, top)

    def to_LayerConcat(self, layer):
        inps = self.get_layers(layer['inputs'])
        inp = inps[0]
        top = np.copy(inp['top'][0])
        for inp in inps[1:]:
            bottom = inp['top'][0]
            top = np.concatenate((top,bottom), axis=layer['axis'])
        layer['top'] = [top]

    def to_LayerSoftmax(self, layer):
        if('axis' in layer):
            axis = layer['axis'] # axis is not handled by lwnn
        else:
            axis = -1
        softmax = torch.nn.Softmax(axis)
        setattr(self, layer.name, softmax)
    def forward_LayerSoftmax(self, layer):
        inp = self.get_layers(layer.inputs)[0]
        bottom = self.get_bottom(layer)
        if('permute' in layer):
            if(len(inp.shape) == 3):
                bottom = torch.reshape(bottom, [inp.shape[i] for i in [0,2,1]])
            else:
                raise NotImplementedError()
        softmax = getattr(self, layer.name)
        top = softmax(bottom)
        self.set_top(layer, top)

    def to_LayerMaxPool(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        with_mask = False
        if(len(layer['outputs']) == 2):
            with_mask = True
        bottom = inp['top'][0]
        top = lwnn.MaxPool2d(bottom,
            kernel_size=LI(layer['kernel_shape']),
            stride=LI(layer['strides']), 
            padding=LI(layer['pads'] if 'pads' in layer else [0,0]),
            output_shape=LI(layer['shape']),
            with_mask=with_mask)
        if(with_mask):
            layer['top'] = top
        else:
            layer['top'] = [top]

    def to_LayerUpsample(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        _,_,Hin,Win = inp['shape']
        _,_,Hout,Wout = layer['shape']
        upsample = torch.nn.Upsample(scale_factor=(int(Hout/Hin), int(Wout/Win)))
        setattr(self, layer.name, upsample)

    def to_LayerBatchNormalization(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        num_features = inp['shape'][1]
        epsilon = self.get_attr(layer, 'epsilon', 1e-5)
        batchnorm = torch.nn.BatchNorm2d(num_features=num_features, eps=epsilon)
        batchnorm.weight = torch.nn.Parameter(torch.from_numpy(layer['scale']))
        batchnorm.bias = torch.nn.Parameter(torch.from_numpy(layer['bias']))
        batchnorm.running_mean = torch.from_numpy(layer['mean'])
        batchnorm.running_var = torch.from_numpy(layer['var'])
        setattr(self, layer.name, batchnorm)

    def to_LayerTranspose(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        perm = layer['perm']
        top = np.transpose(bottom, perm)
        layer['top'] = [top]

    def to_LayerConstant(self, layer):
        setattr(self, layer.name, torch.from_numpy(layer.const))
    def forward_LayerConstant(self, layer):
        self._outputs[layer.outputs[0]] = getattr(self, layer.name)

    def to_LayerGather(self, layer):
        indices = layer['indices']
        setattr(self, layer.name+'_indices', torch.from_numpy(indices.astype(np.int64)))
    def forward_LayerGather(self, layer):
        bottom = self.get_bottom(layer)
        dim = layer['axis']
        top = torch.gather(bottom, dim, getattr(self, layer.name+'_indices'))
        self.set_top(layer, top)

    def to_LayerPriorBox(self, layer):
        pass
    def forward_LayerPriorBox(self, layer):
        feature_shape = np.asarray(layer['feature_shape'], np.int32)
        image_shape = np.asarray(layer['image_shape'], np.int32)
        variance = np.asarray(layer['variance'], np.float32)
        max_sizes = self.get_attr(layer, 'max_size', [])
        if(type(max_sizes) == float):
            max_sizes = [max_sizes]
        max_sizes = np.asarray(max_sizes, np.int32)
        min_sizes = self.get_attr(layer, 'min_size')
        if(type(min_sizes) == float):
            min_sizes = [min_sizes]
        min_sizes = np.asarray(min_sizes, np.int32)
        aspect_ratios = self.get_attr(layer, 'aspect_ratio', 1.0)
        if(type(aspect_ratios) == float):
            aspect_ratios = [aspect_ratios]
        aspect_ratios = np.asarray(aspect_ratios, np.float32)
        flip = self.get_attr(layer, 'flip', 1.0)
        clip = self.get_attr(layer, 'clip', 0.0)
        step = self.get_attr(layer, 'step', 0.0)
        offset = self.get_attr(layer, 'offset', 0.5)
        output_shape = np.asarray(layer['shape'], np.int32)
        top = lwnn.PriorBox(feature_shape, image_shape, variance, max_sizes, min_sizes,
                            aspect_ratios, clip, flip, step, offset, output_shape)
        layer['top'] = [top]

    def to_LayerLSTM(self, layer):
        W,R,B = layer.W,layer.R,layer.B
        I,H,O = W.shape[-1], int(B.shape[-1]/8), R.shape[-1]
        Wi,Wo,Wf,Wc = W.reshape(4,-1,I)
        Ri,Ro,Rf,Rc = R.reshape(4,-1,H)
        Wbi,Wbo,Wbf,Wbc,Rbi,Rbo,Rbf,Rbc = B.reshape(8, -1)
        lstm = torch.nn.LSTM(I, H, batch_first=False)
        lstm.weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.concatenate([Wi, Wf, Wc, Wo], axis=0)))
        lstm.weight_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.concatenate([Ri, Rf, Rc, Ro], axis=0)))
        lstm.bias_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.concatenate([Wbi, Wbf, Wbc, Wbo], axis=0)))
        lstm.bias_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.concatenate([Rbi, Rbf, Rbc, Rbo], axis=0)))
        setattr(self, layer.name, lstm)
    def forward_LayerLSTM(self, layer):
        bottom = self.get_bottom(layer)
        lstm = getattr(self, layer.name)
        # note: prev_c/prev_h shape MUST be (1, 1, H)
        prev_c_n = '%s:prev_c'%(layer.name)
        if(prev_c_n in self._outputs):
            prev_c = self._outputs[prev_c_n]
            prev_h = self._outputs['%s:prev_h'%(layer.name)]
            top, (new_h, new_c) = lstm(bottom, (prev_h, prev_c))
            self._outputs['%s:new_c'%(layer.name)] = new_c
            self._outputs['%s:new_h'%(layer.name)] = new_h
        else:
            top, (_, _) = lstm(bottom)
        self.set_top(layer, top)


    def to_LayerOutput(self, layer):
        pass
    def forward_LayerOutput(self, layer):
        pass

    def forward(self, feed):
        self._outputs = {}
        for k, v in feed.items():
            if(type(v) == np.ndarray):
                v = torch.from_numpy(v)
            self._outputs[k] = v
        for layer in self.lwnn_model:
            fwdfunc = getattr(self, 'forward_Layer%s'%(layer.op), self.forward_LayerCommon)
            #print('forward %s %s %s'%(layer.name, layer.op, fwdfunc.__name__))
            fwdfunc(layer)
            #print('  ->', self.get_top(layer).shape)
        return self._outputs

    def run(self, feeds):
        for feed in feeds:
            onefeed={}
            for n, v in feed.items():
                onefeed[n] = v
            outputs = self(onefeed)
            yield outputs

def lwnn2torch(model, feeds, **kwargs):
    model = Lwnn2Torch(model, **kwargs)
    print(model)
    feeds = LWNNFeeder(feeds, model.inputs, format='NHWC')
    return model.run(feeds)

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert lwnn to pytorch model')
    parser.add_argument('-i', '--input', help='input lwnn.onnx model', type=str, required=True)
    parser.add_argument('-f', '--feeds', help='a json file describe the feeds in dict format, e.g: {"input":["/path/to/input1", "/path/to/input2"]}', type=str, default=None, required=True)
    parser.add_argument('-s', '--shape', help='shapes of some layers', nargs='+', default=None, required=False)
    args = parser.parse_args()
    kwargs = {}
    if((args.shape is not None) and (len(args.shape)%2 == 0)):
        n = int(len(args.shape)/2)
        shapes = {}
        for i in range(n):
            k = args.shape[2*i]
            shape = eval(args.shape[2*i+1])
            shapes[k] = shape
        kwargs['shapes'] = shapes
    lwnn2torch(args.input, args.feeds, **kwargs)
