# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import os
import pickle
import numpy as np
import glob
import liblwnn as lwnn
from lwnn import *
# ref https://pytorch.org/docs/stable/_modules/torch/nn/quantized/functional.html
import torch
from verifyoutput import *

__all__ = ['lwnn2torch']

def LI(a):
    return np.asarray(a, dtype=np.int32)

class Lwnn2Torch():
    def __init__(self, p, **kargs):
        if('debug' in kargs):
            self._debug = kargs['debug']
        else:
            if('YES' == os.getenv('LWNN_DEBUG')):
                self._debug = True
                lwnn.set_log_level(0)
            else:
                self._debug = False
        self._debugQ = True if 'YES' == os.getenv('LWNN_DEBUGQ') else False
        self.RUNL = {
            'Input': self.run_LayerInput,
            'Conv': self.run_LayerConv,
            'Relu': self.run_LayerRelu,
            'Reshape': self.run_LayerReshape,
            'Concat': self.run_LayerConcat,
            'Softmax': self.run_LayerSoftmax,
            'MaxPool': self.run_LayerMaxPool,
            'Upsample': self.run_LayerUpsample,
            'DetectionOutput': self.run_LayerUnknown,
            'Yolo': self.run_LayerUnknown,
            'YoloOutput': self.run_LayerUnknown,
            'BatchNormalization': self.run_LayerBatchNormalization,
            'Transpose': self.run_LayerTranspose,
            'Const': self.run_LayerConst,
            'Gather': self.run_LayerGather,
            'PriorBox': self.run_LayerPriorBox,
            'LSTM': self.run_LayerLSTM,
            'Output': self.run_LayerOutput }
        self.RUNQL = {
            'Input': self.run_QLayerInput,
            'Conv': self.run_QLayerConv,
            'Relu': self.run_LayerQUnknown,
            'Reshape': self.run_LayerQUnknown,
            'Concat': self.run_LayerQUnknown,
            'Softmax': self.run_LayerQUnknown,
            'MaxPool': self.run_LayerQUnknown,
            'Upsample': self.run_LayerQUnknown,
            'DetectionOutput': self.run_LayerQUnknown,
            'Yolo': self.run_LayerQUnknown,
            'YoloOutput': self.run_LayerQUnknown,
            'BatchNormalization': self.run_LayerQUnknown,
            'Output': self.run_QLayerOutput }
        if(type(p) == str):
            self.lwnn_model = self.load(p)
        else:
            self.lwnn_model = p

    def load(self, p):
        try:
            model=pickle.load(open(p,'rb'))
        except:
            # load python2 generated pickle from python3
            model=pickle.load(open(p,'rb'), fix_imports=True, encoding="latin1")
        return model

    @property
    def inputs(self):
        inps = {}
        for ly in self.lwnn_model:
            if(ly['op'] == 'Input'):
                inps[ly['name']] = ly['shape']
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

    def run_LayerInput(self, layer):
        name = layer['name']
        if(self.feeds != None):
            feed = self.feeds[name]
        else:
            feed = np.random.uniform(low=-1,high=1,size=layer['shape']).astype(np.float32)
        layer['top'] = [feed]

    def run_QLayerInput(self, layer):
        bottom = layer['top'][0]
        top = self.quantize_per_tensor(bottom)
        layer['topq'] = [top]

    def activation(self, layer):
        if('activation' in layer):
            activation = layer['activation']
            bottom = layer['top'][0]
            if(activation == 'leaky'):
                act = torch.nn.LeakyReLU(negative_slope=0.1)
            elif(activation == 'Relu'):
                act = torch.nn.ReLU()
            elif(activation == 'linear'):
                return
            top = act(torch.from_numpy(bottom))
            layer['top'] = [top.detach().numpy()]

    def run_LayerConv(self, layer, Q=False):
        inp = self.get_layers(layer['inputs'])[0]
        W = layer['weights']
        if(layer['group'] == 1):
            W = W.transpose(0,3,1,2)
        else:
            W = W.transpose(3,0,1,2)
        B = layer['bias']
        _,Cin,_,_ = inp['shape']
        _,Cout,_,_ = layer['shape']
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
            bottom = inp['top'][0]
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
            top = conv(torch.from_numpy(bottom))
            layer['top'] = [top.detach().numpy()]

    def run_QLayerConv(self, layer):
        return self.run_LayerConv(layer, Q=True)

    def run_LayerRelu(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        relu = torch.nn.ReLU()
        top = relu(torch.from_numpy(bottom))
        layer['top'] = [top.detach().numpy()]

    def run_LayerReshape(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        top = bottom.reshape([-1]+list(layer['shape'])[1:])
        layer['top'] = [top]

    def run_LayerConcat(self, layer):
        inps = self.get_layers(layer['inputs'])
        inp = inps[0]
        top = np.copy(inp['top'][0])
        for inp in inps[1:]:
            bottom = inp['top'][0]
            top = np.concatenate((top,bottom), axis=layer['axis'])
        layer['top'] = [top]

    def run_LayerSoftmax(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        if('axis' in layer):
            axis = layer['axis'] # axis is not handled by lwnn
        else:
            axis = -1
        if('permute' in layer):
            if(len(bottom.shape) == 3):
                bottom = bottom.reshape([bottom.shape[i] for i in [0,2,1]])
            else:
                raise NotImplementedError()
        softmax = torch.nn.Softmax(axis)
        top = softmax(torch.from_numpy(bottom))
        layer['top'] = [top.detach().numpy()]

    def run_LayerMaxPool(self, layer):
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

    def run_LayerUpsample(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        _,_,Hin,Win = inp['shape']
        _,_,Hout,Wout = layer['shape']
        upsample = torch.nn.Upsample(scale_factor=(int(Hout/Hin), int(Wout/Win)))
        top = upsample(torch.from_numpy(bottom))
        layer['top'] = [top.detach().numpy()]

    def run_LayerBatchNormalization(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        num_features = inp['shape'][1]
        epsilon = self.get_attr(layer, 'epsilon', 1e-5)
        batchnorm = torch.nn.BatchNorm2d(num_features=num_features, eps=epsilon)
        batchnorm.weight = torch.nn.Parameter(torch.from_numpy(layer['scale']))
        batchnorm.bias = torch.nn.Parameter(torch.from_numpy(layer['bias']))
        batchnorm.running_mean = torch.from_numpy(layer['mean'])
        batchnorm.running_var = torch.from_numpy(layer['var'])
        top = batchnorm(torch.from_numpy(bottom))
        layer['top'] = [top.detach().numpy()]

    def run_LayerTranspose(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        perm = layer['perm']
        top = np.transpose(bottom, perm)
        layer['top'] = [top]

    def run_LayerConst(self, layer):
        layer['top'] = [layer['const']]

    def run_LayerGather(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        indices = layer['indices']
        dim = layer['axis']
        top = torch.gather(torch.from_numpy(bottom), dim, torch.from_numpy(indices.astype(np.int64)))
        layer['top'] = [top.detach().numpy()]

    def run_LayerPriorBox(self, layer):
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

    def run_LayerLSTM(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        W,R,B = layer.W,layer.R,layer.B
        I,H,O = W.shape[-1], int(B.shape[-1]/8), R.shape[-1]
        Wi,Wo,Wf,Wc = W.reshape(4,-1,I)
        Ri,Ro,Rf,Rc = R.reshape(4,-1,H)
        Wbi,Wbo,Wbf,Wbc,Rbi,Rbo,Rbf,Rbc = B.reshape(8, -1)
        lstm = torch.nn.LSTM(I, H, batch_first=True)
        lstm.weight_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.concatenate([Wi, Wf, Wc, Wo], axis=0)))
        lstm.weight_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.concatenate([Ri, Rf, Rc, Ro], axis=0)))
        lstm.bias_ih_l0 = torch.nn.Parameter(torch.from_numpy(np.concatenate([Wbi, Wbf, Wbc, Wbo], axis=0)))
        lstm.bias_hh_l0 = torch.nn.Parameter(torch.from_numpy(np.concatenate([Rbi, Rbf, Rbc, Rbo], axis=0)))
        top, (_, _) = lstm(torch.from_numpy(bottom))
        layer['top'] = [top.detach().numpy()]

    def run_LayerUnknown(self, layer):
        layer['top'] = [None]

    def run_LayerQUnknown(self, layer):
        layer['topq'] = [None]

    def run_LayerOutput(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['top'][0]
        layer['top'] = [bottom]

    def run_QLayerOutput(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        bottom = inp['topq'][0]
        layer['topq'] = [bottom]

    def debug(self, layer):
        try:
            os.makedirs('./tmp/torch')
        except:
            pass
        name = layer['name']
        top = layer['top']
        op = layer['op']
        if((True == self._debugQ) and (op in self.RUNQL)):
            self.RUNQL[op](layer)
        for id, v in enumerate(top):
            if(v is None):
                continue
            print('  saving %s[%s]: %s'%(name, id, v.shape))
            for batch in range(v.shape[0]):
                oname = './tmp/torch/torch-%s-O%s-B%s.raw'%(name, id, batch)
                B = v[batch]
#                 if(os.path.exists('./tmp/%s.raw'%(name)) and (id == 0) and (batch == 0)):
#                     G = np.fromfile('./tmp/%s.raw'%(name), np.float32)
#                     compare(G, B, name)
                if(len(B.shape) == 3):
                    B = B.transpose(1,2,0)
                B.tofile(oname)
            if((True == self._debugQ) and(op in self.RUNQL)):
                vq = layer['topq'][id]
                if(vq is not None):
                    vfq = torch.nn.quantized.DeQuantize()(vq).detach().numpy()
                    compare(v, vfq, name, tn='torch.q')

    def sanity_check(self, layer):
        top = layer['top'][0]
        if(top is not None):
            eshape = ['?'] + list(layer['shape'])[1:]
            oshape = ['?'] + list(top.shape)[1:]
            if(('permute' in layer) and (layer['op'] == 'Softmax')):
                if(len(eshape) == 3):
                    eshape = [eshape[s] for s in [0,2,1]]
                else:
                    raise NotImplementedError()
            if(eshape != oshape):
                raise Exception('layer %s:\n\texpected shape %s, not %s'%(layer, eshape, oshape))

    def run(self, feeds):
        self.feeds = feeds
        outputs = {}
        for ly in self.lwnn_model:
            if(self._debug):
                print('execute %s %s: %s'%(ly['op'], ly['name'], tuple(ly['shape'])))
            self.RUNL[ly['op']](ly)
            self.activation(ly)
            if(self._debug):
                self.debug(ly)
            self.sanity_check(ly)
            outputs[ly['name']] = ly['top']
        return outputs

def lwnn2torch(model, feeds, **kargs):
    model = Lwnn2Torch(model, **kargs)

    if(type(feeds) == str):
        feeds = load_feeds(feeds, model.inputs)
    return model.run(feeds)

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert lwnn to pytorch model')
    parser.add_argument('-i', '--input', help='input lwnn model', type=str, required=True)
    parser.add_argument('-r', '--raw', help='input raw directory', type=str, default=None, required=True)
    parser.add_argument('-d', '--debug', help='debup outputs of each layer', action='store_true', default=False, required=False)
    args = parser.parse_args()
    lwnn2torch(args.input, args.raw, debug=args.debug)
