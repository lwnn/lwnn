
import onnx
import onnxruntime
import os
import numpy as np

__all__ = ['onnx2lwnn']

class LWNNModel():
    def __init__(self, onnx_model):
        self.TRANSLATOR = {
                    'Transpose': self.to_LayerTranspose,
                    'Conv': self.to_LayerConv,
                    'Identity': self.to_LayerIdentity }
        self.onnx_model = onnx_model
        self.lwnn_model = self.convert()

    def get_inputs(self, node):
        inputs = []
        for inp in self.onnx_model.graph.input:
            if(inp.name in node.input):
                inputs.append(inp.name)
        for node2 in self.onnx_model.graph.node:
            for out in node2.output:
                if(out in node.input):
                    inputs.append(node2.name)
        return inputs

    def eval_outputs(self, node):
        outputs = []
        oldnodes = [n for n in self.onnx_model.graph.node]
        Id = 0
        for n in self.onnx_model.graph.node:
            Id += 1
            if(n == node):
                break
        newnodes = oldnodes[0:Id]
        del self.onnx_model.graph.node[:]
        self.onnx_model.graph.node.extend(newnodes)

        oldoutputs = [n for n in self.onnx_model.graph.output]
        del self.onnx_model.graph.output[:]
        newoutputs = [onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None) 
                        for output in node.output]
        self.onnx_model.graph.output.extend(newoutputs)

        onnx.save(self.onnx_model, '.tmp.onnx')
        del self.onnx_model.graph.node[:]
        self.onnx_model.graph.node.extend(oldnodes)
        del self.onnx_model.graph.output[:]
        self.onnx_model.graph.output.extend(oldoutputs)

        sess = onnxruntime.InferenceSession('.tmp.onnx')
        feed = {}
        for inp in sess.get_inputs():
            shape = list(inp.shape)
            if(shape[0] == None):
                shape[0] = 1
            data = np.random.uniform(low=0,high=1,size=shape).astype(np.float32)
            feed[inp.name] = data
        outputs = sess.run(node.output, feed)

        return outputs

    def get_shape(self, node):
        outputs = self.eval_outputs(node)
        return outputs[0].shape

    def get_initializer(self, name):
        for init in self.onnx_model.graph.initializer:
            if(name == init.name):
                return init
        raise Exception('ERROR: weights %s is not found'%(name))

    def get_layers(self, names):
        layers = []
        for layer in self.lwnn_model:
            if(layer['name'] in names):
                layers.append(layer)
        return layers

    def to_LayerCommon(self, node):
        layer = {'name': node.name, 'op': node.op_type, 'inputs':self.get_inputs(node)}
        layer['shape'] = self.get_shape(node)
        return layer

    def to_LayerTranspose(self, node):
        layer = self.to_LayerCommon(node)
        for attr in node.attribute:
            if(attr.name == 'perm'):
                layer[attr.name] = attr.ints
        return layer

    def to_LayerConv(self, node):
        layer = self.to_LayerCommon(node)
        for attr in node.attribute:
            if(attr.name in ['dilations', 'kernel_shape', 'strides', 'pads']):
                layer[attr.name] = attr.ints
        W = self.get_initializer(node.input[1])
        B = self.get_initializer(node.input[2])
        layer['filters'] = int(W.dims[0])
        layer['weights'] = np.asarray(W.float_data, dtype=np.float32).reshape(W.dims)
        layer['bias'] = np.asarray(B.float_data, dtype=np.float32).reshape(B.dims)
        return layer

    def to_LayerIdentity(self, node):
        layer = self.to_LayerCommon(node)
        return layer

    def convert(self):
        lwnn_model = []
        for inp in self.onnx_model.graph.input:
            shape = [int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim]
            if(shape[0] == 0):
                shape[0] = 1
            layer = {'name': inp.name, 
                     'op': 'Input',
                     'shape': shape }
            lwnn_model.append(layer)
        for node in self.onnx_model.graph.node:
            if(node.op_type in self.TRANSLATOR):
                layer = self.TRANSLATOR[node.op_type](node)
                if(layer != None):
                    lwnn_model.append(layer)
                else:
                    print('WARNINING: layer %s is ignored:\n%s\n'%(node.name, node))
            else:
                raise Exception('ERROR: OP %s is not supported:\n%s\n'%(node.op_type, node))
        for layer in lwnn_model:
            print(layer)
        return lwnn_model

def onnx2lwnn(model, name):
    if('/' not in name):
        p = 'models/%s'%(name)
    else:
        p = name
    if(not p.endswith('.c')):
        p = p + '.c'
    d = os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    print('LWNN %s'%(p))

    if(type(model) == str):
        model = onnx.load(model)
    else:
        with open(p[:-2]+'.onnx','wb') as f:
            f.write(model.SerializeToString())

    model = LWNNModel(model)

    fp = open(p, 'w')
    fp.write('#include "nn.h"')
    
    fp.close()
