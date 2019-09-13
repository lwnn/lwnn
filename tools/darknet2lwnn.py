# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn import *
import os
import configparser
import io
from collections import defaultdict

__all__ = ['dartnet2lwnn']

class DarknetConverter():
    def __init__(self, cfg, weights):
        self.cfgFile = cfg
        unique_config_file = self.unique_config_sections(cfg)
        self.cfg = configparser.ConfigParser()
        self.cfg.read_file(unique_config_file)
        self.lwnn_model = []
        self.TRANSLATOR = {
            'net': self.to_LayerInput,
            'convolutional': self.to_LayerConv,
            'shortcut': self.to_LayerShortcut,
            'upsample': self.to_LayerUpsample,
            'yolo': self.to_LayerYolo,
            'route': self.to_LayerRoute,
             }
        self.opMap = { 
            'convolutional': 'Conv',
            'route': 'Concat'
            }

    def unique_config_sections(self, config_file):
        """Convert all config sections to have unique names.
        Adds unique suffixes to config sections for compability with configparser.
        """
        section_counters = defaultdict(int)
        output_stream = io.StringIO()
        with open(config_file) as fin:
            for line in fin:
                if line.startswith('['):
                    section = line.strip().strip('[]')
                    _section = section + '_' + str(section_counters[section])
                    section_counters[section] += 1
                    line = line.replace(section, _section)
                output_stream.write(line)
        output_stream.seek(0)
        return output_stream

    def save(self, path):
        pass

    def to_LayerInput(self, cfg):
        shape = [1, eval(cfg['channels']), eval(cfg['height']), eval(cfg['width'])]
        layer = { 'name': 'data', 
                  'op': 'Input',
                  'outputs' : ['data'],
                  'shape':  shape }
        return layer

    def to_LayerConv(self, cfg):
        layer = self.to_LayerCommon(cfg)
        stride = layer['stride']
        pad = layer['pad']
        size = layer['size']
        filters = layer['filters']
        layer['strides'] = [stride,stride]
        n,c,h,w = layer['shape']
        if(size != 1):
            h = int(h/stride+pad*2-(size-1))
            w = int(w/stride+pad*2-(size-1))
        layer['shape'] = [n,filters, h, w]
        return layer

    def to_LayerShortcut(self, cfg):
        layer = self.to_LayerCommon(cfg)
        fr = layer['from']
        if(fr < 0):
            inp = self.lwnn_model[fr]
        else:
            inp = self.lwnn_model[fr+1]
        layer['inputs'] = [inp['name']]
        layer['shape'] = inp['shape']
        return layer

    def to_LayerRoute(self, cfg):
        layer = self.to_LayerCommon(cfg)
        lys = layer['layers']
        inputs = []
        try:
            lys = tuple(lys)
        except:
            lys = tuple([lys])
        for id in lys:
            if(id < 0):
                inp = self.lwnn_model[id]
            else:
                inp = self.lwnn_model[id+1]
            inputs.append(inp)
        layer['inputs'] = [L['name'] for L in inputs]
        n,c,h,w = inputs[0]['shape']
        for inp in inputs[1:]:
            c += inp['shape'][1]
        layer['shape'] = [n,c,h,w]
        return layer

    def to_LayerUpsample(self, cfg):
        layer = self.to_LayerCommon(cfg)
        stride = layer['stride']
        n,c,h,w = layer['shape']
        h = h*stride
        w = w*stride
        layer['shape'] = [n, c, h, w]
        return layer

    def to_LayerYolo(self, cfg):
        layer = self.to_LayerCommon(cfg)
        layer['shape'] = []
        return layer

    def to_LayerCommon(self, cfg):
        inp = self.lwnn_model[-1] # generally the last layer is the input
        op = cfg.name.split('_')[0]
        if(op in self.opMap):
            op = self.opMap[op]
        op = op[0].upper() + op[1:]
        layer = { 'name': cfg.name, 
                  'op': op,
                  'outputs' : [cfg.name],
                  'inputs' : [inp['name']],
                  'shape': inp['shape']
                }
        for k,v in cfg.items():
            try:
                v = eval(v)
            except:
                pass
            layer[k] = v
        return layer

    def get_layers(self, names):
        layers = []
        for ly in self.lwnn_model:
            if(ly['name'] in names):
                layers.append(ly)
        return layers

    def __str__(self):
        cstr = 'Darknet Model %s:\n'%(self.cfgFile)
        for ID, L in enumerate(self.lwnn_model):
            cstr += '%3s %-17s: %-18s'%(ID-1, L['name'], L['shape'])
            if(L['op'] == 'Shortcut'):
                cstr += ' %s'%(L['inputs'][0])
            elif(L['op'] == 'Conv'):
                inp = self.get_layers(L['inputs'])[0]
                cstr += ' %-18s %s %sx%s/%s'%(inp['shape'], inp['name'], L['size'], L['size'], L['stride'])
            elif(L['op'] == 'Upsample'):
                cstr += ' x%s'%(L['stride'])
            elif(L['op'] == 'Concat'):
                cstr += ' %s'%(L['inputs'])
            cstr += '\n'
        return cstr

    def convert(self):
        self.lwnn_model = []
        for section in self.cfg.sections():
            op = section.split('_')[0]
            if(op in self.TRANSLATOR):
                translator = self.TRANSLATOR[op]
            else:
                translator = self.to_LayerCommon
            layer = translator(self.cfg[section])
            self.lwnn_model.append(layer)
        return self.lwnn_model

def dartnet2lwnn(cfg, name, **kargs):
    if('weights' in kargs):
        weights = kargs['weights']
    else:
        weights = None
    model = LWNNModel(DarknetConverter(cfg, weights), name)
    if('feeds' in kargs):
        feeds = kargs['feeds']
    else:
        feeds = None
    model.gen_float_c(feeds)


if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert onnx to lwnn')
    parser.add_argument('-i', '--input', help='input dartnet cfg', type=str, required=True)
    parser.add_argument('-w', '--weights', help='input dartnet weights', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    parser.add_argument('-r', '--raw', help='input raw directory', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-9]
    dartnet2lwnn(args.input, args.output, weights=args.weights, feeds=args.raw)
