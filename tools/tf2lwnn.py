# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

from lwnn.core import *
import tensorflow as tf
import numpy as np
import os
import onnx
from onnx.shape_inference import infer_shapes
import tf2onnx
from tf2onnx.tfonnx import process_tf_graph, tf_optimize
from onnx2lwnn import OnnxConverter

try:
    TF_VERSION = eval(str(tf.VERSION).replace('.', ','))
except:
    TF_VERSION = eval(str(tf.__version__).replace('.', ','))

IS_TF_V2 = False
if(TF_VERSION[0] == 2):
    IS_TF_V2 = True
    tfFastGFile = tf.io.gfile.GFile
    tfGraphDef = tf.compat.v1.GraphDef
    tfSession = tf.compat.v1.Session
    raise NotImplementedError('tf version 2 is not supported for now')
else:
    tfFastGFile = tf.gfile.FastGFile
    tfGraphDef = tf.GraphDef
    tfSession = tf.Session

__all__ = ['tf2lwnn']

class TfConverter(LWNNUtil):
    def __init__(self, graph_def, name, feeds=None):
        self.TRANSLATOR = {
            'Reshape': self.to_LayerReshape,
            'DecodeWav': self.to_LayerDecodeWav,
            'MatMul': self.to_LayerMatMul,
            'Add': self.to_LayerAdd,
            'Constant': self.to_LayerConst,
            'Transpose': self.to_LayerTranspose }
        self.opMap = {
            'Placeholder': 'Input',
            'Const': 'Constant',
            'BiasAdd': 'Add',
            }
        if(type(graph_def) == str):
            with tfFastGFile(graph_def, 'rb') as f:
                graph_def = tfGraphDef()
                graph_def.ParseFromString(f.read())
        self.graph_def = graph_def
        self.name = name
        self.feeds = feeds
        self.sess = tfSession()
        _ = tf.import_graph_def(self.graph_def, name=self.name)
        if(IS_TF_V2):
            tf.summary.create_file_writer('./graphs')
        else:
            tf.summary.FileWriter('./graphs', self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
        self.tensors = {}
        for node in self.graph_def.node:
            x = self.sess.graph.get_tensor_by_name('%s/%s:0'%(self.name, node.name))
            self.tensors[node.name] = x

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
        op = node.op
        if(op in self.opMap):
            op = self.opMap[op]
        layer = LWNNLayer(name=node.name, op=op)
        if(self.has_field(node, 'input')):
            inputs = []
            input_ref=[]
            for inp in node.input:
                try:
                    name,id= inp.split(':')
                    inputs.append(name)
                    input_ref.append(eval(id))
                except ValueError:
                    inputs.append(inp)
                    input_ref.append(0)
            if(len(inputs) > 0):
                layer['inputs'] = inputs
                layer['input_ref'] = input_ref
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
                try:
                    attr = np.copy(np.ndarray(
                            shape=shape,
                            dtype=dtype,
                            buffer=tensor.tensor_content))
                except:
                    if(dtype == np.int32):
                        attr = np.asarray(tensor.int_val, dtype=dtype)
                    elif(dtype == np.float32):
                        attr = np.asarray(tensor.float_val, dtype=dtype)
                    else:
                        raise
            else:
                raise NotImplementedError('attr %s=%s of node %s is not supported'%(k, attr, node))
            layer[k] = attr
        tensor = self.tensors[node.name]
        try:
            shape = tensor.shape.as_list()
        except ValueError:
            if((self.feeds != None) and (node.name in self.feeds)):
                shape = list(self.feeds[node.name].shape)
                shape[0] = 1
            else:
                shape = []
        layer['outputs'] = [node.name]
        if(len(shape) > 0):
            if(shape[0] == None):
                shape[0] = 1
            layer['shape'] = shape
        return layer

    def to_LayerReshape(self, layer):
        if('shape' not in layer):
            _, shape = self.get_layers(layer.inputs)
            layer.shape = self.sess.run(self.tensors[shape.name]).tolist()
        layer.inputs = layer.inputs[:1]

    def to_LayerDecodeWav(self, layer):
        layer.outputs.append('%s:1'%(layer.name))

    def to_LayerMatMul(self, layer):
        _, weights = self.get_layers(layer.inputs)
        layer.weights = self.sess.run(self.tensors[weights.name])
        layer.inputs = layer.inputs[:1]

    def to_LayerAdd(self, layer):
        _, bias = self.get_layers(layer.inputs)
        try:
            layer.bias = self.sess.run(self.tensors[bias.name])
            layer.inputs = layer.inputs[:1]
        except:
            pass

    def to_LayerConst(self, layer):
        layer.const = layer.value
        layer.shape = layer.value.shape

    def to_LayerTranspose(self, layer):
        _, perm = self.get_layers(layer.inputs)
        layer.perm = self.sess.run(self.tensors[perm.name]).tolist()
        layer.inputs = layer.inputs[:1]

    def run(self, feed=None, **kwargs):
        pass

    def convert2onnx(self):
        inputs = ['%s:0'%(layer.name) for layer in self.lwnn_model if layer.op == 'Input']
        outputs = ['%s:0'%(layer.name) for layer in self.lwnn_model if len(self.get_consumers(layer)) == 0]
        custom_ops = {}
        extra_opset = []
        graph_def = tf_optimize(inputs, outputs, self.graph_def, True)
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=tf_graph):
            g = process_tf_graph(tf_graph,
                                 continue_on_error=False,
                                 target=None,
                                 opset=None,
                                 custom_op_handlers=custom_ops,
                                 extra_opset=extra_opset,
                                 shape_override=None,
                                 input_names=inputs,
                                 output_names=outputs,
                                 inputs_as_nchw=None)

        onnx_graph = tf2onnx.optimizer.optimize_graph(g)
        model = onnx_graph.make_model(self.name)
        inps = []
        for inp in model.graph.input:
            if(inp.name in inputs):
                shape = [int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim]
                if(len(shape) == 0):
                    layer = self.get_layers(self.LN(inp.name))[0]
                    x = onnx.helper.make_tensor_value_info(inp.name, inp.type.tensor_type.elem_type, layer.shape)
                    inps.append(x)
                else:
                    inps.append(inp)
        outs = []
        for out in model.graph.output:
            if(out.name in inputs):
                shape = [int(dim.dim_value) for dim in out.type.tensor_type.shape.dim]
                if(len(shape) == 0):
                    layer = self.get_layers(self.LN(out.name))[0]
                    x = onnx.helper.make_tensor_value_info(out.name, out.type.tensor_type.elem_type, layer.shape)
                    inps.append(x)
                else:
                    inps.append(out)
        del model.graph.input[:]
        model.graph.input.extend(inps)
        del model.graph.output[:]
        model.graph.output.extend(outs)
        return model

    def convert(self):
        self.lwnn_model = []
        for node in self.graph_def.node:
            try:
                layer = self.to_LayerCommon(node)
            except Exception as e:
                raise Exception('failed to convert node %s: %s'%(node, e))
            self.lwnn_model.append(layer)
        for layer in self.lwnn_model:
            if(layer.op in self.TRANSLATOR):
                self.TRANSLATOR[layer.op](layer)
        self.onnx_model = self.convert2onnx()
        feeds = self.feeds
        if(feeds != None):
            feeds = {}
            for k,v in self.feeds.items():
                feeds['%s:0'%(k)] = v
        shapes = {}
        for node in self.onnx_model.graph.node:
            name = self.LN(node.name)
            layer = self.get_layers([name])
            if(len(layer) > 0):
                layer = layer[0]
                if('shape' in layer):
                    shape = layer.shape
                    shapes[node.name] = shape
        converter = OnnxConverter(self.onnx_model, feeds, shapes=shapes)
        self.lwnn_modelo = converter.convert()
        return self.lwnn_modelo

def tf2lwnn(graph_def, name, feeds=None):
    model = LWNNModel(TfConverter(graph_def, name, feeds), name, notRmIdentity=True)
    model.gen_float_c(feeds)
    if(feeds != None):
        model.gen_quantized_c(feeds)

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert tf to lwnn')
    parser.add_argument('-i', '--input', help='input tf model', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    parser.add_argument('-s', '--shape', help='shapes of some layers', nargs='+', default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-3]
    feeds = None
    if((args.shape is not None) and (len(args.shape)%2 == 0)):
        n = int(len(args.shape)/2)
        feeds = {}
        for i in range(n):
            k = args.shape[2*i]
            shape = eval(args.shape[2*i+1])
            feeds[k] = np.random.uniform(low=-255, high=255, size=shape).astype(np.float32)
    tf2lwnn(args.input, args.output, feeds)
