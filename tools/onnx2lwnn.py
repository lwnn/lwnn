
import onnx
import onnxruntime
import os
import numpy as np

__all__ = ['onnx2lwnn']

class LWNNFloatC():
    def __init__(self, model):
        self.GENL = {
                'Input': self.gen_LayerInput,
                'Conv': self.gen_LayerConv,
                'Identity': self.gen_LayerOutput }
        self.model = model
        self.fpH = self.model.open('float.h')
        self.fpC = self.model.open('float.c')
        self.fpC.write('#include "%s"\n\n'%(os.path.basename(self.fpH.name)))
        self.gen_layers()
        self.gen_models()
        self.model.close(self.fpH)
        self.model.close(self.fpC)

    def get_shape(self, layer):
        shape = layer['shape']
        if(len(shape)==4):
            shape = [shape[i] for i in [0,2,3,1]]
        return shape

    def gen_LayerInput(self, layer):
        shape = self.get_shape(layer)
        self.fpC.write('#define %s_DIMS %s\n'%(layer['name'], 
                            ','.join(['%s'%(s) for s in shape])))
        self.fpC.write('L_INPUT ({0}, {0}_DIMS, L_DT_FLOAT);\n\n'.format(layer['name']))

    def gen_LayerConv(self, layer):
        W = layer['weights']
        if(len(W.shape)==4):
            W = W.transpose(1,2,3,0)
        elif(len(W.shape)==3):
            W = W.transpose(1,2,0)
        B = layer['bias']
        self.fpH.write('static const float %s_W[/*%s*/] = {'%(layer['name'], W.shape))
        self.fpH.write(', '.join(['%s'%(f) for f in W.reshape(-1)]))
        self.fpH.write('};\n')
        self.fpH.write('static const float %s_B[/*%s*/] = {'%(layer['name'], B.shape))
        self.fpH.write(', '.join(['%s'%(f) for f in B.reshape(-1)]))
        self.fpH.write('};\n')

    def gen_LayerOutput(self, layer):
        self.fpC.write('L_OUTPUT ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_layers(self):
        for layer in self.model.lwnn_model:
            self.GENL[layer['op']](layer)

    def gen_models(self):
        pass


class LWNNModel():
    def __init__(self, onnx_model, name):
        self.TRANSLATOR = {
                    'Transpose': self.to_LayerTranspose,
                    'Conv': self.to_LayerConv,
                    'Identity': self.to_LayerIdentity }
        self.name = name
        if(type(onnx_model) == str):
            onnx_model = onnx.load(onnx_model)
        else:
            self.save(onnx_model)
        self.onnx_model = onnx_model
        self.shapes = self.eval_shapes()
        self.lwnn_model = self.convert()
        self.lwnn_model = self.remove_adjust_layer()
        print(self)

    def save(self, onnx_model):
        with open(self.path+'.onnx','wb') as f:
            f.write(onnx_model.SerializeToString())

    @property
    def path(self):
        if('/' not in self.name):
            p = 'models/%s'%(self.name)
        else:
            p = self.name
        d = os.path.dirname(p)
        os.makedirs(d, exist_ok=True)
        return p

    def open(self, fix='.c'):
        if(fix != '.c'):
            fix = '_' + fix
        p = self.path + fix
        print('LWNN %s'%(p))
        fp = open(p, 'w')
        fp.write('#include "nn.h"\n')
        return fp

    def close(self, fp):
        fp.close()

    def gen_float_c(self):
        LWNNFloatC(self)

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

    def run(self, feed=None):
        outputs = {}
        oldoutputs = [n for n in self.onnx_model.graph.output]
        del self.onnx_model.graph.output[:]
        newoutputs = []
        for node in self.onnx_model.graph.node:
            for output in node.output:
                newoutputs.append(onnx.helper.make_tensor_value_info(output, onnx.TensorProto.FLOAT, None))
        self.onnx_model.graph.output.extend(newoutputs)

        onnx.save(self.onnx_model, '.tmp.onnx')
        del self.onnx_model.graph.output[:]
        self.onnx_model.graph.output.extend(oldoutputs)

        sess = onnxruntime.InferenceSession('.tmp.onnx')
        if(feed == None):
            feed = {}
            for inp in sess.get_inputs():
                shape = list(inp.shape)
                if(shape[0] == None):
                    shape[0] = 1
                data = np.random.uniform(low=0,high=1,size=shape).astype(np.float32)
                feed[inp.name] = data
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

    def get_layers(self, names, model=None):
        layers = []
        if(model == None):
            model = self.lwnn_model
        for layer in model:
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
        return lwnn_model

    def is_input_channel_adjusted(self, layer):
        r = False
        if((layer['op'] == 'Transpose')
            and (len(layer['perm']) == 4)
            and (layer['perm'][0] == 0)
            and (layer['perm'][1] == 3)
            and (layer['perm'][2] == 1)
            and (layer['perm'][3] == 2)):
            inputs = self.get_layers(layer['inputs'])
            if(inputs[0]['op'] == 'Input'):
                r = True
        return r

    def is_any_of_inputs_input_channel_adjusted(self, layer):
        r = False
        if(layer['op'] != 'Input'):
            inputs = self.get_layers(layer['inputs'])
            for inp in inputs: 
                if(self.is_input_channel_adjusted(inp)):
                    r = True
        return r

    def is_output_channel_adjusted(self, layer):
        r = False
        if(layer['op'] == 'Identity'):
            inp = self.get_layers(layer['inputs'])[0]
            if((inp['op'] == 'Transpose')
                and (len(inp['perm']) == 4)
                and (inp['perm'][0] == 0)
                and (inp['perm'][1] == 2)
                and (inp['perm'][2] == 3)
                and (inp['perm'][3] == 1)):
                r = True
        return r

    def remove_adjust_layer(self):
        is_model_channel_first = True
        for layer in self.lwnn_model:
            if(self.is_input_channel_adjusted(layer)):
                is_model_channel_first = False
        if(is_model_channel_first):
            model = self.lwnn_model
        else:
            model = []
            # for ONNX models exported from keras, it was maybe channel last
            # so firstly need to strip those input/ouput adjust
            for layer in self.lwnn_model:
                if(self.is_any_of_inputs_input_channel_adjusted(layer)):
                    # previous layer is a adjust layer
                    new_inputs = []
                    inputs = self.get_layers(layer['inputs'])
                    for inp in inputs: 
                        if(self.is_input_channel_adjusted(inp)):
                            inp_inputs = self.get_layers(inp['inputs'])
                            new_inputs.append(inp_inputs[0]['name'])
                        else:
                            new_inputs.append(inp['name'])
                    new_layer = dict(layer)
                    new_layer['inputs'] = new_inputs
                    model.append(new_layer)
                elif(self.is_input_channel_adjusted(layer)):
                    inputs = self.get_layers(layer['inputs'], model)
                    inp = inputs[0]
                    shape = inp['shape']
                    inp['shape'] = [shape[i] for i in [0,3,1,2]]
                elif(self.is_output_channel_adjusted(layer)):
                    inputs = self.get_layers(layer['inputs'], model)
                    inp = inputs[0]
                    model.remove(inp)
                    new_layer = dict(layer)
                    new_layer['inputs'] = inp['inputs']
                    shape = new_layer['shape']
                    new_layer['shape'] = [shape[i] for i in [0, 3, 1, 2]]
                    model.append(new_layer)
                else:
                    model.append(dict(layer))
        return model

    def __str__(self):
        cstr = 'LWNN Model %s:\n'%(self.name)
        for layer in self.lwnn_model:
            cstr += ' {'
            for k,v in layer.items():
                if(k in ['weights','bias']):
                    cstr += '%s: %s, '%(k, v.shape)
                else:
                    cstr += '%s: %s, '%(k,v)
            cstr += '}\n'
        return cstr

def onnx2lwnn(model, name):
    model = LWNNModel(model, name)
    model.gen_float_c()
