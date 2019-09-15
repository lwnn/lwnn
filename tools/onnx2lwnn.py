# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn import *
import onnx
import onnxruntime
import numpy as np

__all__ = ['onnx2lwnn']

class OnnxConverter():
    def __init__(self, onnx_model):
        self.TRANSLATOR = {
                'Conv': self.to_LayerConv,
                'BatchNormalization': self.to_LayerBatchNormalization,
                'MatMul': self.to_LayerMatMul,
                'Resize': self.to_LayerUpsample,
                'Add': self.to_LayerAdd }
        if(type(onnx_model) == str):
            onnx_model = onnx.load(onnx_model)
        self.onnx_model = onnx_model
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
            for inp in self.onnx_model.graph.input:
                if(inp.name == iname):
                    inputs.append(inp.name)
            for node2 in self.onnx_model.graph.node:
                for out in node2.output:
                    if(out == iname):
                        inputs.append(node2.name)
        return inputs

    def eval_node_output_type(self, output):
        # TODO: yes, this sounds stupid, is there anyway better?
        def is_type_okay(oT):
            oldoutputs = [n for n in self.onnx_model.graph.output]
            del self.onnx_model.graph.output[:]
            newoutputs = [onnx.helper.make_tensor_value_info(output, oT, None)]
            self.onnx_model.graph.output.extend(newoutputs)
            onnx.save(self.onnx_model, '.tmp.onnx')
            del self.onnx_model.graph.output[:]
            self.onnx_model.graph.output.extend(oldoutputs)
            try:
                sess = onnxruntime.InferenceSession('.tmp.onnx')
                return True
            except:
                return False
        for oT in [onnx.TensorProto.FLOAT, onnx.TensorProto.INT64, onnx.TensorProto.INT32]:
            if(is_type_okay(oT)):
                return oT
        raise Exception("can't determint output type for %s"%(output))

    def run(self, feed=None):
        outputs = {}
        oldoutputs = [n for n in self.onnx_model.graph.output]
        del self.onnx_model.graph.output[:]
        newoutputs = []
        for node in self.onnx_model.graph.node:
            for output in node.output:
                oT = self.eval_node_output_type(output)
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
        outputs = self.run()
        for name, r in outputs.items():
            shapes[name] = r.shape
        return shapes

    def get_shape(self, node):
        return self.shapes[node.output[0]]

    def get_initializer(self, name):
        for init in self.onnx_model.graph.initializer:
            if(name == init.name):
                return init
        raise Exception('ERROR: weights %s is not found'%(name))

    def get_weights(self, layer, node, wl):
        for id,name in enumerate(wl):
            W = self.get_initializer(node.input[id+1])
            layer[name] = np.asarray(W.float_data, dtype=np.float32).reshape(W.dims)

    def to_LayerCommon(self, node):
        layer = {'name': node.name, 'op': node.op_type, 'inputs':self.get_inputs(node), 'outputs':node.output}
        layer['shape'] = self.get_shape(node)
        for attr in node.attribute:
            layer[attr.name] = onnx.helper.get_attribute_value(attr)
        return layer

    def to_LayerConv(self, node):
        layer = self.to_LayerCommon(node)
        if('pads' not in layer):
            layer['pads'] = [0,0,0,0]
        W = self.get_initializer(node.input[1])
        B = self.get_initializer(node.input[2])
        layer['filters'] = int(W.dims[0])
        layer['weights'] = np.asarray(W.float_data, dtype=np.float32).reshape(W.dims)
        layer['bias'] = np.asarray(B.float_data, dtype=np.float32).reshape(B.dims)
        return layer

    def to_LayerBatchNormalization(self, node):
        layer = self.to_LayerCommon(node)
        self.get_weights(layer, node, ['scale', 'bias', 'mean', 'var'])
        return layer

    def to_LayerMatMul(self, node):
        layer = self.to_LayerCommon(node)
        W = self.get_initializer(node.input[1])
        layer['weights'] = np.asarray(W.float_data, dtype=np.float32).reshape(W.dims)
        return layer

    def to_LayerUpsample(self, node):
        layer = self.to_LayerCommon(node)
        layer['op'] = 'Upsample'
        return layer

    def to_LayerAdd(self, node):
        layer = self.to_LayerCommon(node)
        try:
            B = self.get_initializer(node.input[1])
            layer['bias'] = np.asarray(B.float_data, dtype=np.float32).reshape(B.dims)
        except:
            pass
        return layer

    def convert(self):
        lwnn_model = []
        for inp in self.onnx_model.graph.input:
            shape = [int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim]
            if(shape[0] == 0):
                shape[0] = 1
            layer = {'name': inp.name, 
                     'op': 'Input',
                     'outputs' : [inp.name],
                     'shape': shape }
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
            layer = {'name': out.name,
                     'op': 'Output',
                     'inputs': [inp['name']],
                     'outputs' : [out.name],
                     'shape': inp['shape'] }
            lwnn_model.append(layer)
        return lwnn_model

def onnx2lwnn(model, name, feeds=None):
    '''
    feeds: mainly used to do quantization
    '''
    model = LWNNModel(OnnxConverter(model), name)
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
