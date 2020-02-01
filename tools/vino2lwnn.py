# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

from lwnn import *
import os
#from openvino import inference_engine as IE

try:
    import xml.etree.cElementTree as ET
except ImportError:
    import xml.etree.ElementTree as ET

__all__ = ['vino2lwnn']

class VinoLayer():
    def __init__(self, layer):
        self.xml = layer

    def shape(self):
        shape = []
        for port in self.xml.find('output'):
            for s in port:
                shape.append(eval(s.text))
            break
        return shape

    def outputs(self):
        outputs = []
        for port in self.xml.find('output'):
            outputs.append(port.attrib['id'])
        return outputs

    def inputs(self):
        inputs = []
        if(None != self.xml.find('input')):
            for port in self.xml.find('input'):
                inputs.append(port.attrib['id'])
        return inputs

    def __getattr__(self, n):
        return self.xml.attrib[n]

class VinoConverter():
    def __init__(self, vino_model, vino_weights):
        tree = ET.parse(vino_model)
        self.ir = tree.getroot()
        self.vino_weights = vino_weights
        self.TRANSLATOR = {
            'Input': self.to_LayerInput,
            'Convolution': self.to_LayerConv,
             }
        self.opMap = {
            'ReLU': 'Relu',
            }

    def to_LayerCommon(self, vly, op=None):
        name = str(vly.name)
        layer = { 'name':name,
                  'outputs': vly.outputs(),
                  'inputs': vly.inputs(),
                  'shape': vly.shape()
                }
        if(op == None):
            op = str(vly.type)
        if(op in self.opMap):
            op = self.opMap[op]
        layer['op'] = op
        print(layer)
        return layer

    def to_LayerInput(self, vly):
        layer = self.to_LayerCommon(vly, 'Input')
        return layer

    def to_LayerConv(self, vly):
        layer = self.to_LayerCommon(vly, 'Conv')
        return layer

    def save(self, path):
        pass

    def run(self, feed):
        outputs = {}
        return outputs

    def convert(self):
        lwnn_model = []
        self.bin = open(self.vino_weights, 'rb')
        print(self.ir.attrib['name'])
        for node in self.ir:
            if(node.tag == 'layers'):
                for layer in node:
                    layer = VinoLayer(layer)
                    op = layer.type
                    if(op in self.TRANSLATOR):
                        translator = self.TRANSLATOR[op]
                    else:
                        translator = self.to_LayerCommon
                    layer = translator(layer)
                    lwnn_model.append(layer)
            else:
                print('TODO: %s'%(node.tag))
        self.bin.close()
        return lwnn_model

    @property
    def inputs(self):
        L = {}
        return L

def vino2lwnn(model, name, **kargs):
    if('weights' in kargs):
        weights = kargs['weights']
    else:
        weights = None
    model = LWNNModel(VinoConverter(model, weights), name)
    if('feeds' in kargs):
        feeds = kargs['feeds']
    else:
        feeds = None

    if(type(feeds) == str):
        feeds = load_feeds(feeds, model.converter.inputs)

    model.gen_float_c(feeds)
    if(feeds != None):
        model.gen_quantized_c(feeds)

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert OpenVINO IR model to lwnn')
    parser.add_argument('-i', '--input', help='input vino IR model (*.xml)', type=str, required=True)
    parser.add_argument('-w', '--weights', help='input vino IR weights (*.bin)', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    parser.add_argument('-r', '--raw', help='input raw directory', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-4]
    vino2lwnn(args.input, args.output, weights=args.weights, feeds=args.raw)
