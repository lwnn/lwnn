# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn.core import *
import onnx
from onnx.shape_inference import infer_shapes
import onnxruntime
import numpy as np

__all__ = ['onnx2lwnn']

class OnnxConverter():
    def __init__(self, onnx_model, feeds=None):
        self.TRANSLATOR = {
                'Conv': self.to_LayerConv,
                'ConvTranspose': self.to_LayerConvTranspose,
                'BatchNormalization': self.to_LayerBatchNormalization,
                'MatMul': self.to_LayerMatMul,
                'Resize': self.to_LayerUpsample,
                'Reshape': self.to_LayerReshape,
                'Constant': self.to_LayerConstant,
                'Add': self.to_LayerAdd }
        if(type(onnx_model) == str):
            onnx_model = onnx.load(onnx_model)
        self.onnx_model = onnx_model
        self.feeds = feeds
        self.shapes = self.eval_shapes()

    @property
    def input(self):
        return self.onnx_model.graph.input
    @property
    def output(self):
        return self.onnx_model.graph.output

    def save(self, path):
        with open(path+'.onnx','wb') as f:
            f.write(self.onnx_model.SerializeToString())

    def get_inputs(self, node):
        inputs = []
        # order is important for some layers such as Concat
        for iname in node.input:
            try:
                _ = self.get_initializer(iname)
                continue
            except:
                pass
            for inp in self.onnx_model.graph.input:
                if(inp.name == iname):
                    inputs.append(inp.name)
            for node2 in self.onnx_model.graph.node:
                for out in node2.output:
                    if(out == iname):
                        if(node2.op_type == 'Identity'):
                            inputs.append(node2.input[0]+'_identity')
                        else:
                            inputs.append(node2.name)
        return inputs

    def run(self, feed=None, **kwargs):
        model2 = infer_shapes(self.onnx_model)
        onnx.save(model2, 'tmp.onnx')
        output_types = {}
        for vinfo in list(model2.graph.value_info) + \
                     list(model2.graph.output) + \
                     list(model2.graph.input):
            output_types[vinfo.name] = vinfo.type.tensor_type.elem_type
        outputs = {}
        oldoutputs = [n for n in self.onnx_model.graph.output]
        del self.onnx_model.graph.output[:]
        newoutputs = []
        for node in self.onnx_model.graph.node:
            for output in node.output:
                if(output in output_types):
                    oT = output_types[output]
                else:
                    oT = onnx.TensorProto.FLOAT
                newoutputs.append(onnx.helper.make_tensor_value_info(output, oT, None))
        self.onnx_model.graph.output.extend(newoutputs)

        onnx.save(self.onnx_model, '.tmp.onnx')
        del self.onnx_model.graph.output[:]
        self.onnx_model.graph.output.extend(oldoutputs)

        sess = onnxruntime.InferenceSession('.tmp.onnx')
        if(feed == None):
            feed = {}
            for inp in sess.get_inputs():
                shape = list(inp.shape)
                if((str(shape[0]) == 'None') or (shape[0] == 'N')):
                    shape[0] = 1
                print(inp, shape)
                data = np.random.uniform(low=-1,high=1,size=shape).astype(np.float32)
                feed[inp.name] = data
        else:
            feedo = feed
            feed = {}
            for inp in sess.get_inputs():
                for name, data in feedo.items():
                    m = min(len(inp.name), len(name))
                    if((inp.name[:m] == name[:m]) or
                       (inp.name[:m-2] == name[:m-2])):
                        feed[inp.name] = data
        for n, v in feed.items():
            outputs[n] = v
        rs = sess.run(None, feed)
        for r,o in zip(rs, newoutputs):
            outputs[o.name] = r

        return outputs

    def eval_shapes(self):
        shapes = {}
        if(self.feeds is None):
            feed = None
        else:
            feed = {}
            for k,v in self.feeds.items():
                feed[k] = v[:1]
        outputs = self.run(feed)
        for name, r in outputs.items():
            shapes[name] = r.shape
        return shapes

    def get_shape(self, node):
        return self.shapes[node.output[0]]

    def tensor2numpy(self, tensor):
        if(tensor.data_type == onnx.TensorProto.FLOAT):
            dtype, data = np.float32, tensor.float_data
        elif(tensor.data_type == onnx.TensorProto.INT32):
            dtype, data = np.int32, tensor.int32_data
        elif(tensor.data_type == onnx.TensorProto.INT64):
            dtype, data = np.int64, tensor.int64_data
        else:
            raise NotImplemented('Type %s not supported'%(tensor.data_type))
        if(len(data) > 0):
            array = np.asarray(data, dtype=np.float32).reshape(tensor.dims)
        else:
            array = np.ndarray(
                                shape=tensor.dims,
                                dtype=dtype,
                                buffer=tensor.raw_data)
        return np.copy(array)

    def get_initializer(self, name):
        for init in self.onnx_model.graph.initializer:
            if(name == init.name):
                return self.tensor2numpy(init)
        raise Exception('ERROR: weights %s is not found'%(name))

    def get_weights(self, layer, node, wl):
        for id,name in enumerate(wl):
            layer[name] = self.get_initializer(node.input[id+1])

    def to_LayerCommon(self, node):
        if(node.op_type == 'Identity'):
            name = node.input[0]+'_identity'
        else:
            name = node.name
        layer = LWNNLayer(name=name, op=node.op_type, inputs=self.get_inputs(node), outputs=node.output)
        layer['shape'] = self.get_shape(node)
        for attr in node.attribute:
            v = onnx.helper.get_attribute_value(attr)
            if(type(v) == onnx.TensorProto):
                v = self.tensor2numpy(v)
            layer[attr.name] = v
        return layer

    def to_LayerConv(self, node):
        layer = self.to_LayerCommon(node)
        if('pads' not in layer):
            layer['pads'] = [0,0,0,0]
        W = self.get_initializer(node.input[1])
        layer['weights'] = W
        if(len(node.input) > 2):
            layer['bias'] = self.get_initializer(node.input[2])
        else:
            nF = W.shape[0]
            layer['bias'] = np.zeros((nF), np.float32)
        layer['filters'] = int(W.shape[0])
        return layer

    def to_LayerConvTranspose(self, node):
        return self.to_LayerConv(node)

    def to_LayerBatchNormalization(self, node):
        layer = self.to_LayerCommon(node)
        self.get_weights(layer, node, ['scale', 'bias', 'mean', 'var'])
        return layer

    def to_LayerMatMul(self, node):
        layer = self.to_LayerCommon(node)
        layer['weights'] = self.get_initializer(node.input[1])
        return layer

    def to_LayerUpsample(self, node):
        layer = self.to_LayerCommon(node)
        layer['op'] = 'Upsample'
        return layer

    def to_LayerConstant(self, node):
        layer = self.to_LayerCommon(node)
        layer['const'] = layer['value']
        del layer['value']
        return layer

    def to_LayerReshape(self, node):
        layer = self.to_LayerCommon(node)
        layer['inputs'] = layer['inputs'][:1]
        return layer

    def to_LayerAdd(self, node):
        layer = self.to_LayerCommon(node)
        try:
            layer['bias'] = self.get_initializer(node.input[1])
        except:
            pass
        return layer

    def convert(self):
        lwnn_model = []
        for inp in self.onnx_model.graph.input:
            shape = [int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim]
            if(len(shape) == 0):
                continue
            if(shape[0] == 0):
                shape[0] = 1
            layer = LWNNLayer(name=inp.name, 
                     op='Input',
                     outputs=[inp.name],
                     shape=shape)
            lwnn_model.append(layer)
        for node in self.onnx_model.graph.node:
            if(node.op_type in self.TRANSLATOR):
                translator = self.TRANSLATOR[node.op_type]
            else:
                translator = self.to_LayerCommon
            layer = translator(node)
            if(layer != None):
                lwnn_model.append(layer)
            else:
                print('WARNINING: layer %s is ignored:\n%s\n'%(node.name, node))
        for out in self.onnx_model.graph.output:
            inp = None
            for ly in lwnn_model:
                if(out.name in ly['outputs']):
                    inp = ly
                    break
            layer = LWNNLayer(name=out.name+'_output',
                     op='Output',
                     inputs=[inp['name']],
                     outputs=[out.name],
                     shape=inp['shape'])
            lwnn_model.append(layer)
        return lwnn_model

def onnx2lwnn(model, name, feeds=None):
    '''
    feeds: mainly used to do quantization
    '''
    model = LWNNModel(OnnxConverter(model, feeds), name)
    model.gen_float_c(feeds)
    if(feeds != None):
        model.gen_quantized_c(feeds)


if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert onnx to lwnn')
    parser.add_argument('-i', '--input', help='input onnx model', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-5]
    onnx2lwnn(args.input, args.output)
