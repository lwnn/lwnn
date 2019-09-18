# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn import *
import os
import configparser
import io
from collections import defaultdict
import numpy as np
#import struct

__all__ = ['dartnet2lwnn']

class DarknetConverter():
    def __init__(self, cfg, weights):
        self.cfgFile = cfg
        self.weightsFile = weights
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
            'route': 'Concat',
            'shortcut': 'Add',
            }

    def read(self, num, type='f'):
        sz = 4
        tM = { 'f':np.float32, 'i':np.int32, 'q':np.int64, 'd':np.float64 }
        if(type in ['q','d']): # long long or double
            sz = 8
        #v = np.array(struct.unpack('<'+str(num)+type, self.weights.read(sz*num)))
        v = np.ndarray(
                shape=(num,),
                dtype=tM[type],
                buffer=self.weights.read(num*sz))
        if(type == 'f'):
            v = v.astype(np.float32)
        return v

    def check_version(self):
        major = self.read(1, 'i')[0]
        minor = self.read(1, 'i')[0]
        revision = self.read(1, 'i')[0]
        if (((major*10 + minor) >= 2) and (major < 1000) and (minor < 1000)):
            seen = self.read(1, 'q')[0]
        else:
            seen = self.read(1, 'i')[0]
        print('Loading weights from %s..., version=%s.%s.%s, seen=%s'%(self.weightsFile, major, minor, revision, seen))
        self.transpose = (major > 1000) or (minor > 1000)

    def unique_config_sections(self, config_file):
        """Convert all config sections to have unique names.
        Adds unique suffixes to config sections for compability with configparser.
        """
        # taken from https://github.com/allanzelener/YAD2K/blob/master/yad2k.py
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
        if(size == 1):
            pad = 0
        filters = layer['filters']
        layer['strides'] = [stride,stride]
        layer['pads'] = [pad, pad, pad, pad]
        if('groups' not in layer):
            group = 1
        else:
            group = layer['groups']
        layer['group'] = group
        n,c,h,w = layer['shape']
        if(size != 1):
            h = int((h+pad*2-size)/stride)+1
            w = int((w+pad*2-size)/stride)+1
        layer['shape'] = [n,filters, h, w]
        layer['bias'] = self.read(filters, 'f')
        if(('batch_normalize' in layer) and 
           (layer['batch_normalize'] == 1)):
            layer['scales'] = self.read(filters, 'f')
            layer['rolling_mean'] = self.read(filters, 'f')
            layer['rolling_variance'] = self.read(filters, 'f')
        layer['weights'] = self.read(int(c/group*filters*size*size), 'f').reshape(filters, c, size, size)
        if(('batch_normalize' in layer) and 
           (layer['batch_normalize'] == 1)):
            c_w = layer['weights']
            c_b = layer['bias']
            bn_gamma = layer['scales']
            bn_mean = layer['rolling_mean']
            bn_variance = layer['rolling_variance']
            epsilon = 0.000001
            if(len(c_w.shape) == 4):
                for i in range(c_w.shape[0]):
                    c_w[i] *= bn_gamma[i] / (np.sqrt(bn_variance[i]) + epsilon)
                    c_b[i] = c_b[i] - bn_gamma[i]*bn_mean[i] / (np.sqrt(bn_variance[i]) + epsilon)
                layer['weights'] = c_w
                layer['bias'] = c_b
            else:
                raise Exception("don't know how to fuse for %s shape %s"%(layer['name'], c_w.shape))
        return layer

    def to_LayerShortcut(self, cfg):
        layer = self.to_LayerCommon(cfg)
        fr = layer['from']
        assert(fr != -1)
        if(fr < 0):
            inp = self.lwnn_model[fr]
        else:
            inp = self.lwnn_model[fr+1]
        inp2 = self.lwnn_model[-1]
        layer['inputs'] = [inp['name'], inp2['name']]
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
        layer['axis'] = 1
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
        total = layer['num']
        mask = layer['mask']
        num = len(mask)
        classes = layer['classes']
        n,c,h,w = layer['shape']
        assert(c == num*(classes + 4 + 1))
        self.yolos.append(layer)
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
            cstr += '%3s %-17s: %-18s %s'%(ID-1, L['name'], L['shape'], L['inputs'] if 'inputs' in L else '')
            if(L['op'] == 'Conv'):
                inp = self.get_layers(L['inputs'])[0]
                cstr += ' %-18s %sx%s/%s'%(inp['shape'], L['size'], L['size'], L['stride'])
            elif(L['op'] == 'Upsample'):
                cstr += ' x%s'%(L['stride'])
            cstr += '\n'
        return cstr

    def setup_output(self):
        if(0 == len(self.yolos)):
            inp = self.lwnn_model[-1]
            oname = inp['name']
            layer = { 'name': oname+'_O', 
                      'op': 'Output',
                      'inputs' : [oname],
                      'outputs' : [oname+'_O'],
                      'shape': inp['shape'] }
        else:
            classes = self.yolos[0]['classes']
            n = self.lwnn_model[0]['shape'][0]
            layer = { 'name': 'YoloOutput', 
                      'op': 'YoloOutput',
                      'inputs' : [L['name'] for L in self.yolos],
                      'outputs' : ['YoloOutput'],
                      'shape': [n, 7 , 1, classes*2],
                      'Output': True }
        self.lwnn_model.append(layer)

    def convert(self):
        self.yolos = []
        self.weights = open(self.weightsFile, 'rb')
        self.check_version()
        self.lwnn_model = []
        for section in self.cfg.sections():
            op = section.split('_')[0]
            if(op in self.TRANSLATOR):
                translator = self.TRANSLATOR[op]
            else:
                translator = self.to_LayerCommon
            layer = translator(self.cfg[section])
            self.lwnn_model.append(layer)
        self.setup_output()
        anymore = self.weights.read()
        if(len(anymore) != 0):
            raise Exception('weights %s mismatched with the cfg %s'%(self.weightsFile, self.cfgFile))
        self.weights.close()
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
