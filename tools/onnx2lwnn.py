
import onnx
import onnxruntime
import os

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

def get_outpus(node, model):
    outputs = []
    content = onnx_model.SerializeToString()
    sess = onnxruntime.InferenceSession(content)
    feed = dict([(input.name, x[n]) for input in sess.get_inputs()])
    pred_onnx = sess.run(None, feed)
    return outputs

def to_lwnn_LayerCommon(node, model):
    layer = {'name': node.name, 'op': node.op_type, 'inputs':get_inputs(node, model)}
    return layer

def to_lwnn_Transpose(node, model):
    layer = to_lwnn_LayerCommon(node, model)
    for attr in node.attribute:
        if(attr.name == 'perm'):
            layer[attr.name] = attr.ints
    return layer

def to_lwnn_Conv(node, model):
    layer = to_lwnn_LayerCommon(node, model)
    for attr in node.attribute:
        if(attr.name in ['dilations', 'kernel_shape', 'strides']):
            layer[attr.name] = attr.ints
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
