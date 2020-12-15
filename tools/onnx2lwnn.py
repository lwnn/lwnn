# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn.core import *
import onnx
from onnx.shape_inference import infer_shapes
import onnxruntime
import numpy as np

__all__ = ['onnx2lwnn', 'OnnxConverter', 'cstr']

class OnnxConverter(LWNNUtil):
    def __init__(self, onnx_model, feeds=None, **kwargs):
        self.TRANSLATOR = {
                'Conv': self.to_LayerConv,
                'ConvTranspose': self.to_LayerConvTranspose,
                'BatchNormalization': self.to_LayerBatchNormalization,
                'MatMul': self.to_LayerMatMul,
                'Resize': self.to_LayerUpsample,
                'Upsample': self.to_LayerUpsample,
                'Reshape': self.to_LayerReshape,
                'Constant': self.to_LayerConstant,
                'Gemm': self.to_LayerGemm,
                'Dense': self.to_LayerGemm,
                'LSTM': self.to_LayerLSTM,
                'Clip': self.to_LayerClip,
                'Transpose': self.to_LayerTranspose,
                'ReduceMean': self.to_LayerReduceMean,
                'Add': self.to_LayerAdd }
        if(type(onnx_model) == str):
            onnx_model = onnx.load(onnx_model)
        self.onnx_model = onnx_model
        self.feeds = feeds
        self.kwargs = kwargs
        self._outputs = {}
        self.shapes = self.eval_shapes()
        self.convert()

    @property
    def model(self):
        return self.lwnn_model

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
        indexs = []
        # order is important for some layers such as Concat
        for iname in node.input:
            if(iname in self.initializers):
                inputs.append(iname)
                indexs.append(0)
            for inp in self.onnx_model.graph.input:
                if(inp.name == iname):
                    inputs.append(inp.name)
            for node2 in self.onnx_model.graph.node:
                for index,out in enumerate(node2.output):
                    if(out == iname):
                        indexs.append(index)
                        if(node2.op_type == 'Identity'):
                            inputs.append(node2.input[0]+'_identity')
                        else:
                            inputs.append(node2.name if len(node2.name) > 0 else node2.output[0])
        return inputs, indexs

    def run(self, feed=None, **kwargs):
        model2 = infer_shapes(self.onnx_model)
        #onnx.save(model2, 'tmp.onnx')
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
                    _opts = list(newoutputs)
                    for oT in [onnx.TensorProto.INT32, onnx.TensorProto.INT64, onnx.TensorProto.FLOAT, onnx.TensorProto.DOUBLE]:
                        vinfo = onnx.helper.make_tensor_value_info(output, oT, None)
                        self.onnx_model.graph.output.extend(_opts+[vinfo])
                        try:
                            sess = onnxruntime.InferenceSession(self.onnx_model.SerializeToString())
                            del self.onnx_model.graph.output[:]
                            break
                        except:
                            del self.onnx_model.graph.output[:]
                newoutputs.append(onnx.helper.make_tensor_value_info(output, oT, None))
        self.onnx_model.graph.output.extend(newoutputs)
        sess = onnxruntime.InferenceSession(self.onnx_model.SerializeToString())
        del self.onnx_model.graph.output[:]
        self.onnx_model.graph.output.extend(oldoutputs)
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

    def infer_node_shape(self, node, inputs):
        if(node.op_type == 'Gemm'):
            bias = self.get_initializer(node.input[2])
            shape = list(inputs[0].shape)[:-2] + list(bias.shape)
        elif(len(inputs) == 1):
            x = onnx.helper.make_tensor_value_info(node.input[0], onnx.TensorProto.FLOAT, inputs[0].shape)
            outputs = [onnx.helper.make_tensor_value_info(o, onnx.TensorProto.FLOAT, None) for o in node.output]
            graph = onnx.helper.make_graph(
                nodes = [node],
                name = 'infer-node-shape',
                inputs = [x],
                outputs = outputs,
                value_info = [],
                initializer = self.onnx_model.graph.initializer)
            model = onnx.helper.make_model(graph, producer_name='lwnn-nhwc')
            model2 = infer_shapes(model)
            shape = [int(dim.dim_value) for dim in model2.graph.output[0].type.tensor_type.shape.dim]
        else:
            raise NotImplementedError('infer %s(%s) shape is not supported'%(node.name, node.op_type))
        if(len(shape) == 0):
            raise Exception("can't infer %s(%s) shape"%(node.name, node.op_type))
        return shape

    def eval_shapes(self):
        if('shapes' in self.kwargs):
            return self.kwargs['shapes']
        shapes = {}
        if(('lwnn' in self.kwargs) and (self.kwargs['lwnn'] == True)):
            for vinfo in self.onnx_model.graph.value_info:
                shapes[vinfo.name] = [int(dim.dim_value) for dim in vinfo.type.tensor_type.shape.dim]
            return shapes
        if(self.feeds is None):
            feed = None
        else:
            feed = {}
            for k,v in self.feeds.items():
                feed[k] = v[:1]
        outputs = self.run(feed)
        for name, r in outputs.items():
            shapes[name] = r.shape
        self._outputs = outputs
        return shapes

    def get_shape(self, node):
        name = node.output[0]
        if(name in self.shapes):
            return self.shapes[name]
        name = self.LN(name)
        if(name in self.shapes):
            return self.shapes[name]
        inputs = self.get_layers([self.LN(n) for n in node.input])
        return self.infer_node_shape(node, inputs)

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
        inits = []
        for init in self.onnx_model.graph.initializer:
            if(name == init.name):
                inits.append(self.tensor2numpy(init))
        if(len(inits) > 0):
            return inits[-1]
        raise Exception('ERROR: weights %s is not found'%(name))

    def get_weights(self, layer, wl, offset=1):
        for id,name in enumerate(wl):
            layer[name] = self.initializers[layer.inputs[id+offset]]

    def to_LayerCommon(self, node):
        if(node.op_type == 'Identity'):
            name = node.input[0]+'_identity'
        else:
            name = node.name
        if(len(name) == 0):
            name = node.output[0]
        inputs,indexs=self.get_inputs(node)
        layer = LWNNLayer(name=name, op=node.op_type, inputs=inputs, input_indexs=indexs, outputs=node.output)
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
        W = self.initializers[node.input[1]]
        layer['weights'] = W
        if(len(node.input) > 2):
            layer['bias'] = self.initializers[node.input[2]]
        else:
            nF = W.shape[0]
            layer['bias'] = np.zeros((nF), np.float32)
        layer.inputs = layer.inputs[:1]
        return layer

    def to_LayerConvTranspose(self, node):
        return self.to_LayerConv(node)

    def to_LayerBatchNormalization(self, node):
        layer = self.to_LayerCommon(node)
        self.get_weights(layer, ['scale', 'bias', 'mean', 'var'])
        layer.inputs = layer.inputs[:1]
        return layer

    def to_LayerMatMul(self, node):
        layer = self.to_LayerCommon(node)
        if(layer.inputs[1] in self.initializers):
            layer.weights = self.initializers[layer.inputs[1]]
        return layer

    def to_LayerUpsample(self, node):
        layer = self.to_LayerCommon(node)
        layer.inputs = layer.inputs[:1]
        layer['op'] = 'Upsample'
        return layer

    def to_LayerConstant(self, node):
        layer = self.to_LayerCommon(node)
        layer['const'] = layer['value']
        del layer['value']
        del layer['inptuts']
        return layer

    def to_LayerReshape(self, node):
        layer = self.to_LayerCommon(node)
        layer['inputs'] = layer['inputs'][:1]
        return layer

    def to_LayerAdd(self, node):
        layer = self.to_LayerCommon(node)
        if(layer.inputs[1] in self.initializers):
            layer['bias'] = self.initializers[layer.inputs[1]]
            layer.inputs = layer.inputs[:1]
        return layer

    def to_LayerGemm(self, node):
        layer = self.to_LayerCommon(node)
        layer['weights'] = self.initializers[layer.inputs[1]]
        layer['bias'] = self.initializers[layer.inputs[2]]
        layer.inputs = layer.inputs[:1]
        layer['op'] = 'Dense'
        return layer

    def to_LayerLSTM(self, node):
        layer = self.to_LayerCommon(node)
        layer['W'] = self.initializers[layer.inputs[1]]
        layer['R'] = self.initializers[layer.inputs[2]]
        if(len(layer.inputs) > 3):
            layer['B'] = self.initializers[layer.inputs[3]]
        if(len(layer.shape) == 4):
            layer.shape = [layer.shape[s] for s in [0,2,1,3]]
        self.convert_layer_to_nchw(layer)
        self.convert_layer_to_nchw(self.get_layers(layer.inputs[0]))
        layer.inputs = layer.inputs[:1]
        return layer

    def to_LayerClip(self, node):
        layer = self.to_LayerCommon(node)
        layer.min = self.initializers[layer.inputs[1]]
        if(len(layer.inputs) > 2):
            layer.max = self.initializers[layer.inputs[2]]
        else:
            if(layer.min == 0.0):
                layer.op = 'Relu'
        layer.inputs = layer.inputs[:1]
        return layer

    def to_LayerTranspose(self, node):
        layer = self.to_LayerCommon(node)
        if(layer.inputs[0] in self.initializers):
            # okay transpose a const input
            data = self.initializers[layer.inputs[0]]
            layer.const = np.transpose(data, layer.perm)
            layer.op = 'Constant'
            del layer['inputs']
        return layer

    def to_LayerReduceMean(self, node):
        layer = self.to_LayerCommon(node)
        layer.axis = layer.axes[0]
        del layer['axes']
        return layer

    def convert(self):
        self.lwnn_model = []
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
            self.lwnn_model.append(layer)
        self.initializers = {}
        for init in self.onnx_model.graph.initializer:
            if(init.name not in self.initializers):
                v = self.get_initializer(init.name)
                self.initializers[init.name] = v
                layer = LWNNLayer(name=init.name, 
                     op='Constant',
                     const = v,
                     outputs=[init.name],
                     shape=v.shape)
                self.lwnn_model.append(layer)
        for node in self.onnx_model.graph.node:
            if(node.op_type in self.TRANSLATOR):
                translator = self.TRANSLATOR[node.op_type]
            else:
                translator = self.to_LayerCommon
            layer = translator(node)
            if(layer != None):
                self.lwnn_model.append(layer)
            else:
                print('WARNINING: layer %s is ignored:\n%s\n'%(node.name, node))
        for out in self.onnx_model.graph.output:
            inp = None
            for ly in self.lwnn_model:
                if(out.name in ly['outputs']):
                    inp = ly
                    break
            layer = LWNNLayer(name=out.name+'_output',
                     op='Output',
                     inputs=[inp['name']],
                     outputs=[out.name],
                     shape=inp['shape'])
            self.lwnn_model.append(layer)
        for layer in self.lwnn_model:
            if('inpust' in layer):
                L = self.get_layers(layer.inputs)
                if(all(l.op == 'Constant' for l in L)):
                    if(len(layer.outputs) == 1):
                        if(layer.outputs[0] in self._outputs):
                            v = self._outputs[layer.outputs[0]]
                        else:
                            continue
                        layer.const = v
                        layer.op = 'Constant'
                        del layer['inputs']

def onnx2lwnn(model, name, feeds=None):
    '''
    feeds: mainly used to do quantization
    '''
    model = LWNNModel(OnnxConverter(model, feeds), name, feeds=feeds)
    model.generate()


if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert onnx to lwnn')
    parser.add_argument('-i', '--input', help='input onnx model', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-5]
    onnx2lwnn(args.input, args.output)
