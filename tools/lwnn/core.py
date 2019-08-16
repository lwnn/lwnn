# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from .float import *
from .qformat import *

class LWNNModel():
    def __init__(self, onnx_model, name):
        self.TRANSLATOR = {
                    'Conv': self.to_LayerConv,
                    'BatchNormalization': self.to_LayerBatchNormalization,
                    'MatMul': self.to_LayerMatMul,
                    'Add': self.to_LayerAdd }
        self.OPTIMIER = [
            (self.opt_IsLayerUnused, self.opt_LayerUnusedAction, None),
            (self.opt_IsLayerBeforeReshape, self.opt_LayerBeforeReshape, None),
            (self.opt_IsLayerDense, self.opt_LayerDense, None),
            (self.opt_IsLayerConv1D, self.opt_LayerConv1D, None),
            (self.opt_IsLayerMaxPool1D, self.opt_LayerMaxPool1D, None),
            (self.opt_IsLayerConvBeforeBN, self.opt_FuseConvBN, None),
            (self.opt_IsLayerConv, self.opt_LayerConvWeightsReorder, None),
            (self.opt_IsLayerReshape, self.opt_RemoveReshape, 'RemoveReshape'),
            ]
        self.toNCHW = [
            (self.nchw_IsPreviousHasInputAdjustLayer, self.nchw_ActionPreviousHasInputAdjustLayer),
            (self.nchw_IsInputAdjustLayer, self.nchw_ActionInputAdjustLayer),
            (self.nchw_IsPreviousHasOutputAdjustLayer, self.nchw_ActionPreviousHasOutputAdjustLayer),
            (self.nchw_IsIdentityChannelAdjusted, self.nchw_ActionIdentityChannelAdjusted),
            (self.nchw_IsOkay, self.nchw_ActionOkay)
            ]
        self.is_model_channel_first_cached=None
        self.name = name
        if(type(onnx_model) == str):
            onnx_model = onnx.load(onnx_model)
        else:
            self.save(onnx_model)
        self.onnx_model = onnx_model
        self.shapes = self.eval_shapes()
        self.lwnn_model = self.convert()
        self.lwnn_model = self.convert_to_nchw()
        #intentionally do it twice
        self.lwnn_model = self.convert_to_nchw()
        self.optimize()
        self.check()
        print(self)

    def clone(self):
        return [dict(ly) for ly in self.lwnn_model]

    def set(self, model):
        self.lwnn_model = model

    def save(self, onnx_model):
        with open(self.path+'.onnx','wb') as f:
            f.write(onnx_model.SerializeToString())

    @property
    def path(self):
        if('/' not in self.name):
            p = 'models/%s/%s'%(self.name,self.name)
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

    def gen_quantized_c(self, feeds):
        LWNNQFormatC(self, 'q8', feeds)
        LWNNQFormatC(self, 'q16', feeds)

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
                if(shape[0] == None):
                    shape[0] = 1
                data = np.random.uniform(low=-1,high=1,size=shape).astype(np.float32)
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

    def get_layers(self, names, model=None):
        layers = []
        if(model == None):
            model = self.lwnn_model
        for layer in model:
            if(layer['name'] in names):
                layers.append(layer)
        return layers

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

    def to_LayerAdd(self, node):
        layer = self.to_LayerCommon(node)
        B = self.get_initializer(node.input[1])
        layer['bias'] = np.asarray(B.float_data, dtype=np.float32).reshape(B.dims)
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
        return lwnn_model

    def nchw_IsInputAdjustLayer(self, layer):
        r = False
        if((layer['op'] == 'Transpose')
            and ((list(layer['perm']) == [0, 3, 1, 2]) or 
                  (list(layer['perm']) == [0, 2, 1]))):
            if(self.is_model_channel_first_cached == None):
                # yes, don't to be too aggressive
                inp = self.get_layers(layer['inputs'])[0]
                if(inp['op'] == 'Input'):
                    r = True
                else:
                    # resnet50 case: Input->Pad->Transpose
                    inp_inputs = self.get_layers(inp['inputs'])
                    for l in inp_inputs:
                        if(l['op'] == 'Input'):
                            r = True
            else:
                if(list(layer['perm']) == [0, 2, 1]):
                    inp = self.get_layers(layer['inputs'])[0]
                    if(inp['op'] == 'Input'):
                        r = True
                else:
                    r = True
        return r

    def convert_axis_to_nchw(self, layer):
        axis = layer['axis']
        shape = layer['shape']
        if(len(shape) == 4):
            axis = [0,2,3,1][axis]
        if(len(shape) == 3):
            axis = [0,2,1][axis]
        layer['axis'] = axis

    def convert_layer_to_nchw(self, layer):
        shape = layer['shape']
        if('adjusted' not in layer):
            if(len(shape) == 4):
                layer['shape'] = [shape[i] for i in [0,3,1,2]]
            if(len(shape) == 3):
                layer['shape'] = [shape[i] for i in [0,2,1]]
            if(layer['op'] == 'Concat'):
                self.convert_axis_to_nchw(layer)
            layer['adjusted'] = True

    def nchw_ActionInputAdjustLayer(self, layer, model):
        inputs = self.get_layers(layer['inputs'], model)
        if(len(inputs) == 0):
            inputs = self.get_layers(layer['inputs'])
            inp = inputs[0]
        else:
            inp = inputs[0]
        if(inp['op'] == 'Pad'):
            inp_inputs = self.get_layers(inp['inputs'], model)
            ly = inp_inputs[0]
            self.convert_layer_to_nchw(ly)
        self.convert_layer_to_nchw(inp)
        return []

    def nchw_IsPreviousHasInputAdjustLayer(self, layer):
        r = False
        if(layer['op'] != 'Input'):
            inputs = self.get_layers(layer['inputs'])
            for inp in inputs: 
                if(self.nchw_IsInputAdjustLayer(inp)):
                    r = True
        return r

    def nchw_ActionPreviousHasInputAdjustLayer(self, layer, model):
        new_inputs = []
        inputs = self.get_layers(layer['inputs'])
        for inp in inputs: 
            if(self.nchw_IsInputAdjustLayer(inp)):
                inp_inputs = self.get_layers(inp['inputs'])
                for ly in inp_inputs:
                    if(self.nchw_IsIdentityChannelAdjusted(ly)):
                        new_inputs.append(self.get_layers(ly['inputs'])[0]['name'])
                    else:
                        new_inputs.append(ly['name'])
            else:
                new_inputs.append(inp['name'])
        new_layer = dict(layer)
        new_layer['inputs'] = new_inputs
        return [new_layer]

    def nchw_IsOutputAdjustLayer(self, layer):
        r = False
        if((layer['op'] == 'Transpose')
            and ((list(layer['perm']) == [0, 2, 3, 1]) or 
                  (list(layer['perm']) == [0, 2, 1]))):
            r = True
        return r

    def nchw_IsPreviousHasOutputAdjustLayer(self, layer):
        r = False
        if(layer['op'] != 'Input'):
            inputs = self.get_layers(layer['inputs'])
            for inp in inputs:
                if(self.nchw_IsOutputAdjustLayer(inp)):
                    r = True
        return r

    def nchw_ActionPreviousHasOutputAdjustLayer(self, layer, model):
        new_inputs = []
        inputs = self.get_layers(layer['inputs'], model)
        for inp in inputs:
            if(self.nchw_IsOutputAdjustLayer(inp)):
                inp_inputs = self.get_layers(inp['inputs'])
                for ly in inp_inputs:
                    if(self.nchw_IsIdentityChannelAdjusted(ly)):
                        new_inputs.append(self.get_layers(ly['inputs'])[0]['name'])
                    else:
                        new_inputs.append(ly['name'])
                model.remove(inp)
            else:
                new_inputs.append(inp['name'])
        new_layer = dict(layer)
        new_layer['inputs'] = new_inputs
        self.convert_layer_to_nchw(new_layer)
        return [new_layer]

    def nchw_IsIdentityChannelAdjusted(self, layer):
        r = False
        if(layer['op'] == 'Identity'):
            consumers = self.get_consumers(layer)
            for ly in consumers: 
                if(self.nchw_IsInputAdjustLayer(ly)):
                    r = True
        return r

    def nchw_ActionIdentityChannelAdjusted(self, layer, model):
        return self.nchw_ActionInputAdjustLayer(layer, model)

    def is_model_channel_first(self):
        if(self.is_model_channel_first_cached != None):
            return self.is_model_channel_first_cached
        r = True
        for layer in self.lwnn_model:
            if(self.nchw_IsInputAdjustLayer(layer)):
                r = False
        self.is_model_channel_first_cached = r
        return r

    def nchw_IsOkay(self, layer):
        return True

    def nchw_ActionOkay(self, layer, model):
        new_layer = dict(layer)
        if(new_layer['op'] == 'Identity'):
            self.convert_layer_to_nchw(new_layer)
        return [new_layer]

    def convert_to_nchw(self):
        if(self.is_model_channel_first()):
            model = self.lwnn_model
        else:
            model = []
            # for ONNX models exported from keras, it was maybe channel last
            # so firstly need to strip those input/ouput adjust,
            # convert the model to format NCHW
            for layer in self.lwnn_model:
                for cond, act in self.toNCHW:
                    if(cond(layer)):
                        model.extend(act(layer, model))
                        break
        return model

    def get_consumers(self, layer):
        consumers = []
        for ly in self.lwnn_model:
            if('inputs' not in ly): continue
            if(layer['name'] in ly['inputs']):
                consumers.append(ly)
        return consumers

    def is_there_op(self, layers, op):
        for ly in layers:
            if(ly['op'] == op):
                return True
        return False

    def get_between_layers(self, fr, to):
        layers = [fr]
        stop = False
        while(fr != to):
            consumers = self.get_consumers(fr)
            if(len(consumers) == 1):
                fr = consumers[0]
                if(fr != to):
                    layers.append(fr)
            else:
                raise NotImplementedError()
        return layers

    def insert_after(self, after, layer):
        for id,ly in enumerate(self.lwnn_model):
            if(ly == after):
                self.lwnn_model.insert(id, layer)
                break

    def opt_IsLayerConvBeforeBN(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((layer['op'] == 'Conv') and 
               (len(consumers) == 1) and
               (consumers[0]['op'] == 'BatchNormalization')):
            r = True
        return r

    def opt_FuseConvBN(self, layer):
        consumers = self.get_consumers(layer)
        bn = consumers[0]
        c_w = layer['weights']
        c_b = layer['bias']
        bn_gamma = bn['scale']
        bn_beta = bn['bias']
        bn_mean = bn['mean']
        bn_variance = bn['var']
        epsilon = bn['epsilon']
        if(len(c_w.shape) == 4):
            for i in range(c_w.shape[0]):
                c_w[i] *= bn_gamma[i] / np.sqrt(bn_variance[i] + epsilon)
                c_b[i] = (bn_gamma[i] * (c_b[i] - bn_mean[i]) / np.sqrt(bn_variance[i] + epsilon)) + bn_beta[i]
        layer['weights'] = c_w
        layer['bias'] = c_b
        bn_consumers = self.get_consumers(bn)
        for ly in bn_consumers:
            new_ly = dict(ly)
            new_ly['inputs'] = bn['inputs']
            self.insert_after(bn, new_ly)
        for ly in [bn] + bn_consumers:
            self.lwnn_model.remove(ly)
        return True

    def opt_IsLayerBeforeReshape(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((len(consumers) == 2) and
           self.is_there_op(consumers, 'Reshape')):
            r = True
        return r

    def opt_LayerBeforeReshape(self, layer):
        consumers = self.get_consumers(layer)
        if(consumers[0]['op'] == 'Reshape'):
            to = consumers[0]
            fr = consumers[1]
        else:
            to = consumers[1]
            fr = consumers[0]

        layers = self.get_between_layers(fr, to)
        new_layer = dict(to)
        new_layer['inputs'] = layer['inputs']
        self.insert_after(to, new_layer)
        for ly in [layer,to]+layers:
            if(ly['op'] == 'Concat'):
                inputs = self.get_layers(ly['inputs'])
                for inp in inputs:
                    if(inp in self.lwnn_model):
                        self.lwnn_model.remove(inp)
            self.lwnn_model.remove(ly)
        return True

    def opt_IsLayerDense(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((layer['op'] == 'MatMul') and 
               (len(consumers) == 1) and
               (consumers[0]['op'] == 'Add')):
            r = True
        return r

    def opt_LayerDense(self, layer):
        consumers = self.get_consumers(layer)
        add = consumers[0]
        new_layer = dict(add)
        new_layer['op'] = 'Dense'
        new_layer['inputs'] = layer['inputs']
        new_layer['weights'] = layer['weights']
        self.insert_after(add, new_layer)
        for ly in [layer,add]:
            self.lwnn_model.remove(ly)
        return True

    def opt_IsLayerConv1D(self, layer):
        r = False
        if((layer['op'] == 'Conv') and 
            (len(layer['weights'].shape) == 3)):
            r = True
        return r

    def opt_LayerConv1D(self, layer):
        W = layer['weights']
        shape = list(W.shape) + [1]
        W = W.reshape(shape)
        layer['weights'] = W
        strides = layer['strides']
        layer['strides'] = list(strides)+ [1]
        pads = layer['pads']
        layer['pads'] = [pads[0], 0, pads[1], 0]
        kernel_shape = layer['kernel_shape']
        layer['kernel_shape'] = list(kernel_shape)+ [1]
        return False

    def opt_IsLayerMaxPool1D(self, layer):
        r = False
        if((layer['op'] == 'MaxPool') and 
            (len(layer['kernel_shape']) == 1)):
            r = True
        return r

    def opt_LayerMaxPool1D(self, layer):
        strides = layer['strides']
        layer['strides'] = list(strides)+ [1]
        if('pads' in layer):
            pads = layer['pads']
            layer['pads'] = list(pads)+ [1]
        kernel_shape = layer['kernel_shape']
        layer['kernel_shape'] = list(kernel_shape)+ [1]
        return False

    def opt_IsLayerConv(self, layer):
        r = False
        if(layer['op'] == 'Conv'):
            r = True
        return r

    def opt_LayerConvWeightsReorder(self, layer):
        # Conv: [M x C/group x kH x kW] -> [M x kH x kW x C/group]
        # DwConv: [M x C/group x kH x kW] -> [C/group x kH x kW x M]
        W = layer['weights']
        if(len(W.shape)==4):
            W = W.transpose(0,2,3,1)
        if(layer['group'] == 1):
            pass
        elif(layer['group'] == layer['shape'][1]):
            if(len(W.shape)==4):
                W = W.transpose(3,1,2,0)
        layer['weights'] = W
        return False

    def opt_IsLayerReshape(self, layer):
        r = False
        if(layer['op'] == 'Reshape'):
            r = True
        return r

    def opt_RemoveReshape(self, layer):
        consumers = self.get_consumers(layer)
        for ly in consumers:
            new_layer = dict(ly)
            new_layer['inputs'] = layer['inputs']
            self.insert_after(layer, new_layer)
        for ly in [layer] + consumers:
            self.lwnn_model.remove(ly)
        return True

    def opt_IsLayerUnused(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((len(consumers) == 0) and (layer['op'] != 'Identity')):
            r = True
        return r

    def opt_LayerUnusedAction(self, layer):
        self.lwnn_model.remove(layer)
        return True

    def optimize(self, additions=[]):
        id = 0
        num_layers = len(self.lwnn_model)
        while(id < (num_layers-1)):
            id += 1
            layer = self.lwnn_model[id]
            for isopt, optact, oname in self.OPTIMIER:
                if(isopt(layer) and
                   (((oname == None) and (len(additions) == 0)) 
                    or (oname in additions))):
                    r = optact(layer)
                    if(True == r): # if there is remove action, restart optimization
                        id = 0
                        num_layers = len(self.lwnn_model)
                        break

    def check(self):
        for id,layer in enumerate(self.lwnn_model):
            if('inputs' in layer):
                # check that inputs are before me
                inputs = self.get_layers(layer['inputs'],self.lwnn_model[:id])
                if(len(inputs) != len(layer['inputs'])):
                    raise Exception('layer %s inputs is not before me:\n%s'%(layer['name'], self))

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
