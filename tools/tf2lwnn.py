# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

from lwnn.core import *
import tensorflow as tf
import numpy as np
import os

__all__ = ['tf2lwnn']

class TfConverter():
    def __init__(self, graph_def, feeds=None):
        self.TRANSLATOR = {
            'Placeholder': self.to_LayerInput }
        if(type(graph_def) == str):
            with tf.gfile.FastGFile(graph_def, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
        self.graph_def = graph_def
        self.feeds = feeds

    def save(self, path):
        pass

    def has_field(self, attr, field):
        try:
            return attr.HasField(field)
        except ValueError:
            try:
                _ = attr.__getattribute__(field)
                return True
            except ValueError:
                False

    def to_LayerCommon(self, node):
        layer = LWNNLayer(name=node.name, op=node.op)
        if(self.has_field(node, 'input')):
            inputs = [inp for inp in node.input]
            if(len(inputs) > 0):
                layer['inputs'] = inputs
        for k in node.attr:
            attr = node.attr[k]
            if(self.has_field(attr,'type')):
                if('DT_STRING' in str(attr)):
                    attr = 'string'
                elif('DT_INT32' in str(attr)):
                    attr = 'int32'
                elif('DT_FLOAT' in str(attr)):
                    attr = 'float'
                elif('DT_RESOURCE' in str(attr)):
                    attr = 'resource' 
                else:
                    raise NotImplementedError('type %s of node %s is not supported'%(attr, node))
            elif(self.has_field(attr,'shape')):
                attr = [int(dim) for dim in attr.shape.dim]
            elif(self.has_field(attr,'i')):
                attr = attr.i
            elif(self.has_field(attr,'b')):
                attr = attr.b
            elif(self.has_field(attr,'f')):
                attr = attr.f
            elif(self.has_field(attr,'s')):
                attr = attr.s
            elif(self.has_field(attr,'list')):
                L = attr.list
                if(self.has_field(L,'s')):
                    attr = [s.decode('utf-8') for s in L.s]
                else:
                    raise
            elif(self.has_field(attr,'tensor')):
                tensor = attr.tensor
                shape = [dim.size for dim in tensor.tensor_shape.dim]
                if('DT_INT32' in str(tensor)):
                    dtype = np.int32
                elif('DT_INT64' in str(tensor)):
                    dtype = np.int64
                elif('DT_FLOAT' in str(tensor)):
                    dtype = np.float32 
                else:
                    raise NotImplementedError('type of tensor %s is not supported'%(attr))
                if(self.has_field(tensor,'int_val')):
                    attr = np.ndarray(tensor.int_val, dtype=dtype)
                else:
                    attr = np.copy(np.ndarray(
                            shape=shape,
                            dtype=dtype,
                            buffer=tensor.tensor_content))
            else:
                raise NotImplementedError('attr %s=%s of node %s is not supported'%(k, attr, node))
            layer[k] = attr
        layer['outputs'] = [node.name]
        return layer

    def to_LayerInput(self, node):
        layer = self.to_LayerCommon(node)
        return layer

    def convert(self):
        self.lwnn_model = []
        for node in self.graph_def.node:
            if(node.op in self.TRANSLATOR):
                translator = self.TRANSLATOR[node.op]
            else:
                translator = self.to_LayerCommon
            try:
                layer = translator(node)
            except Exception as e:
                raise Exception('failed to convert node %s: %s'%(node, e))
            if(layer != None):
                self.lwnn_model.append(layer)
        return self.lwnn_model

def tf2lwnn(graph_def, name, feeds=None):
    model = LWNNModel(TfConverter(graph_def, feeds), name)
    model.gen_float_c(feeds)
    if(feeds != None):
        model.gen_quantized_c(feeds)

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert tf to lwnn')
    parser.add_argument('-i', '--input', help='input tf model', type=str, required=True)
    parser.add_argument('-o', '--output', help='output tf model', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-3]
    tf2lwnn(args.input, args.output)
