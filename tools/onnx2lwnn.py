
import onnx
import onnxruntime
import os
import numpy as np

__all__ = ['onnx2lwnn']

def get_inputs(node, model):
    inputs = []
    for inp in model.graph.input:
        if(inp.name in node.input):
            inputs.append(inp.name)
    for node2 in model.graph.node:
        for out in node2.output:
            if(out in node.input):
                inputs.append(node2.name)
    return inputs

def eval_outputs(node, model):
    outputs = []
    oldnodes = [n for n in model.graph.node]
    Id = 0
    for n in model.graph.node:
        Id += 1
        if(n == node):
            break
    newnodes = oldnodes[0:Id]
    del model.graph.node[:]
    model.graph.node.extend(newnodes)

    oldoutputs = [n for n in model.graph.output]
    print(oldoutputs)
    del model.graph.output[:]
    newoutputs = [node]
    model.graph.output.extend(newoutputs)
    
    onnx.save(model, '.tmp.onnx')
    del model.graph.node[:]
    model.graph.node.extend(oldnodes)
    del model.graph.output[:]
    model.graph.output.extend(oldoutputs)

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

def get_shape(node, model):
    outputs = eval_outputs(node, model)
    return outputs[0].shape
    
def to_lwnn_LayerCommon(node, model):
    layer = {'name': node.name, 'op': node.op_type, 'inputs':get_inputs(node, model)}
    #layer['shape'] = get_shape(node, model)
    return layer

def to_lwnn_Transpose(node, model):
    layer = to_lwnn_LayerCommon(node, model)
    for attr in node.attribute:
        if(attr.name == 'perm'):
            layer[attr.name] = attr.ints
    return layer

def get_initializer(name, model):
    for init in model.graph.initializer:
        if(name == init.name):
            return init

def to_lwnn_Conv(node, model):
    layer = to_lwnn_LayerCommon(node, model)
    for attr in node.attribute:
        if(attr.name in ['dilations', 'kernel_shape', 'strides']):
            layer[attr.name] = attr.ints
    W = get_initializer(node.input[1], model)
    B = get_initializer(node.input[2], model)
    layer['filters'] = int(W.dims[0])
    layer['weights'] = np.asarray(W.float_data, dtype=np.float32).reshape(W.dims)
    layer['bias'] = np.asarray(B.float_data, dtype=np.float32).reshape(B.dims)
    return layer

def to_lwnn_Identity(node, model):
    layer = to_lwnn_LayerCommon(node, model)
    return layer

TRANSLATOR = {'Transpose': to_lwnn_Transpose,
              'Conv': to_lwnn_Conv,
              'Identity': to_lwnn_Identity }

def to_lwnn_model(model):
    lwnn_model = []
    for inp in model.graph.input:
        shape = [int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim]
        if(shape[0] == 0):
            shape[0] = 1
        layer = {'name': inp.name, 
                 'op': 'Input',
                 'shape': shape }
        lwnn_model.append(layer)
    for node in model.graph.node:
        if(node.op_type in TRANSLATOR):
            layer = TRANSLATOR[node.op_type](node, model)
            if(layer != None):
                lwnn_model.append(layer)
            else:
                print('WARNINING: layer %s is ignored:\n%s\n'%(node.name, node))
        else:
            raise Exception('ERROR: OP %s is not supported:\n%s\n'%(node.op_type, node))
    for layer in lwnn_model:
        print(layer)
    return lwnn_model

def gen_weights_for_each_layer(layer, fp):
    print(layer.name)

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

    model = to_lwnn_model(model)

    fp = open(p, 'w')
    fp.write('#include "nn.h"')
    
    fp.close()
