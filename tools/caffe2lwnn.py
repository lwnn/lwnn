# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn import *
import os
os.environ['GLOG_minloglevel'] = '2'
import caffe
from caffe.proto import caffe_pb2
from google.protobuf import text_format

__all__ = ['caffe2lwnn']

class CaffeConverter():
    def __init__(self, caffe_model, caffe_weights):
        self.net = caffe_pb2.NetParameter()
        text_format.Merge(open(caffe_model,'r').read(),self.net)
        self.model = caffe.Net(caffe_model,caffe_weights,caffe.TEST)
        if(len(self.net.layer)==0):    #some prototxts use "layer", some use "layers"
            self.layers = self.net.layers
        else:
            self.layers = self.net.layer
        self.TRANSLATOR = {
            'Convolution': self.to_LayerConv,
             }
        self.opMap = {'ReLU': 'Relu'}

    def to_LayerCommon(self, cly, op=None):
        name = str(cly.name)
        blob = self.model.blobs[cly.top[0]]
        layer = { 'name':name,
                  'outputs': [str(o) for o in cly.top],
                  'inputs': [str(o) for o in cly.bottom],
                  'shape': blob.data.shape
                }
        if(op == None):
            op = str(cly.type)
        if(op in self.opMap):
            op = self.opMap[op]
        layer['op'] = op 
        return layer

    def to_LayerInput(self, cly):
        layer = self.to_LayerCommon(cly, 'Input')
        return layer

    def to_LayerConv(self, cly):
        layer = self.to_LayerCommon(cly, 'Conv')
        name = layer['name']
        params = self.model.params[name]
        layer['weights'] = params[0].data
        layer['bias'] = params[1].data
        stride = cly.convolution_param.stride
        pad = cly.convolution_param.pad
        if(len(stride)==1):
            strideW  = strideH = eval(str(stride[0]))
        else:
            strideW  = strideH = 1
        if(len(pad)==1):
            padW  = padH = eval(str(pad[0]))
        else:
            padW  = padH = 1
        layer['strides'] = [strideW,strideH]
        layer['pads'] = [padW,padH,0,0]
        layer['group'] = 1
        return layer

    def save(self, path):
        pass

    def run(self, feed=None):
        outputs = {}
        return outputs

    def get_layers(self, names, lwnn_model):
        layers = []
        for ly in lwnn_model:
            if(ly['name'] in names):
                layers.append(ly)
        return layers

    def get_inputs(self, layer, lwnn_model):
        inputs = []
        for iname in layer['inputs']:
            for ly in lwnn_model:
                for o in ly['outputs']:
                    if(o == iname):
                        inputs.append(ly['name'])
                        if(len(inputs) == len(layer['inputs'])):
                            # caffe may reuse top buffer to save memory
                            return inputs
        return inputs

    def convert(self):
        lwnn_model = []
        for ly in self.layers:
            op = str(ly.type)
            if(op in self.TRANSLATOR):
                translator = self.TRANSLATOR[op]
            else:
                translator = self.to_LayerCommon
            layer = translator(ly)
            lwnn_model.append(layer)
        for ly in self.model.inputs:
            layers = self.get_layers([ly], lwnn_model)
            if(len(layers) == 0):
                layer = { 'name': ly, 
                          'op': 'Input',
                          'outputs' : [ly],
                          'shape': self.model.blobs[ly].data.shape }
                lwnn_model.insert(0, layer)
        for ly in self.model.outputs:
            layer = { 'name': ly+'_O', 
                      'op': 'Output',
                      'inputs' : [ly],
                      'outputs' : [ly],
                      'shape': self.model.blobs[ly].data.shape }
            lwnn_model.append(layer)
        for ly in lwnn_model:
            if('inputs' in ly):
                inputs = self.get_inputs(ly, lwnn_model)
                if(len(inputs)):
                    ly['inputs'] = inputs
                else:
                    print('xx',ly['name'], ly['inputs'])
                    exit()
        return lwnn_model

def caffe2lwnn(model, name, **kargs):
    if('weights' in kargs):
        weights = kargs['weights']
    else:
        weights = None
    model = LWNNModel(CaffeConverter(model, weights), name)
    if('feeds' in kargs):
        feeds = kargs['feeds']
    else:
        feeds = None

    model.gen_float_c(feeds)
    if(feeds != None):
        model.gen_quantized_c(feeds)

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert onnx to lwnn')
    parser.add_argument('-i', '--input', help='input caffe model', type=str, required=True)
    parser.add_argument('-w', '--weights', help='input caffe weights', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-9]
    caffe2lwnn(args.input, args.output, weights=args.weights)
