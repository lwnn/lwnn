# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from .float import *
from .qformat import *

class LWNNModel():
    def __init__(self, converter, name):
        self.OPTIMIER = [
            (self.nchw_IsLayerNHWC, self.nchw_ActionLayerNHWC, None),
            (self.nchw_IsInputAdjustLayer, self.nchw_ActionInputAdjustLayer, None),
            (self.nchw_IsOutputAdjustLayer, self.opt_RemoveLayer, None),
            (self.opt_IsLayerUnused, self.opt_LayerUnusedAction, None),
            (self.opt_IsLayerBeforeReshape, self.opt_LayerBeforeReshape, None),
            (self.opt_IsLayerDense, self.opt_LayerDense, None),
            (self.opt_IsLayerConv1D, self.opt_LayerConv1D, None),
            (self.opt_IsLayerPooling1D, self.opt_LayerPooling1D, None),
            (self.opt_IsLayerConvBeforeBN, self.opt_FuseConvBN, None),
            (self.opt_IsLayerConv, self.opt_LayerConvWeightsReorder, None),
            (self.opt_IsTrainingOperators, self.opt_RemoveLayer, None),
            (self.opt_IsLayerTransposeCanBeRemoved, self.opt_RemoveLayer, None),
            (self.opt_IsLayerConcatOnPriorBox, self.opt_ReplaceAsConstant, None),
            (self.opt_IsLayerDetectionOutputWithConst, self.opt_MergeConstToDetectionOutput, None),
            (self.opt_IsLayerReshapeBeforeSoftmax, self.opt_PermuteReshapeSoftmax, None),
            (self.opt_IsLayerOutputWithOutput, self.opt_RemoveOutputWithOutput, None),
            (self.opt_IsLayerIdentity, self.opt_RemoveLayer, 'RemoveIdentity'),
            (self.opt_IsLayerReshape, self.opt_RemoveLayer, 'RemoveReshape'),
            (self.opt_IsLayerReLUConv, self.opt_MergeReLUConv, 'MergeReLUConv'),
            (self.opt_IsLayerReLUDense, self.opt_MergeReLUDense, 'MergeReLUDense'),
            ]
        self.is_model_channel_first_cached=None
        self.converter = converter
        self.name = name
        self.converter.save(self.path)
        self.lwnn_model = self.converter.convert()
        # optimization and convert to NCHW if origin model is NHWC
        self.prepare()
        self.omodel = self.clone()
        self.optimize(['RemoveIdentity'])
        self.omodel = self.clone()
        self.optimize()
        self.check()
        print(self)

    @property
    def input(self):
        return self.converter.input
    @property
    def output(self):
        return self.converter.output

    def run(self, feed=None):
        return self.converter.run(feed)

    def clone(self):
        return [dict(ly) for ly in self.lwnn_model]

    def set(self, model):
        self.lwnn_model = model

    @property
    def path(self):
        if('/' not in self.name):
            p = 'models/%s/%s'%(self.name,self.name)
        else:
            p = self.name
        d = os.path.dirname(p)
        try:
            os.makedirs(d)
        except:
            if(not os.path.exists(d)):
                raise Exception('Fatal Error: can\'t create directory <%s>'%(d))
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

    def gen_float_c(self, feeds=None):
        LWNNFloatC(self, feeds)

    def gen_quantized_c(self, feeds):
        LWNNQFormatC(self, 'q8', feeds)
        LWNNQFormatC(self, 'q16', feeds)
        LWNNQSFormatC(self, feeds)

    def get_layers(self, names, model=None):
        layers = []
        if(model == None):
            model = self.lwnn_model
        for layer in model:
            if(layer['name'] in names):
                layers.append(layer)
        return layers

    def nchw_IsInputAdjustLayer(self, layer):
        r = False
        if(self.is_model_channel_first_cached==True):
            pass
        elif((layer['op'] == 'Transpose')
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

    def nchw_ActionInputAdjustLayer(self, layer):
        inputs = self.get_layers(layer['inputs'])
        for inp in inputs:
            if(inp['op'] == 'Pad'):
                inp_inputs = self.get_layers(inp['inputs'])
                for ly in inp_inputs:
                    self.convert_layer_to_nchw(ly)
        return self.opt_RemoveLayer(layer);

    def nchw_IsConsumerHasInputAdjustLayer(self, layer):
        r = False
        consumers = self.get_consumers(layer, self.omodel)
        for ly in consumers:
            if(self.nchw_IsInputAdjustLayer(ly)):
                r = True
        return r

    def nchw_IsOutputAdjustLayer(self, layer):
        r = False
        if(self.is_model_channel_first_cached==True):
            pass
        elif((layer['op'] == 'Transpose')
            and ((list(layer['perm']) == [0, 2, 3, 1]) or 
                  (list(layer['perm']) == [0, 2, 1]))):
            r = True
        return r

    def nchw_IsPreviousHasOutputAdjustLayer(self, layer):
        r = False
        layer = self.get_layers([layer['name']], self.omodel)[0]
        if(layer['op'] != 'Input'):
            inputs = self.get_layers(layer['inputs'], self.omodel)
            for inp in inputs:
                if(self.nchw_IsOutputAdjustLayer(inp)):
                    r = True
        return r

    def is_model_channel_first(self):
        if(self.is_model_channel_first_cached != None):
            return self.is_model_channel_first_cached
        r = True
        for layer in self.lwnn_model:
            if(self.nchw_IsInputAdjustLayer(layer)):
                r = False
        self.is_model_channel_first_cached = r
        return r

    def nchw_IsLayerNHWC(self, layer):
        r = False
        if(self.is_model_channel_first_cached==True):
            pass
        elif(layer['op'] not in ['Conv', 'MaxPool', 'AveragePool']):
            CHIA = self.nchw_IsConsumerHasInputAdjustLayer(layer)
            PHOA = self.nchw_IsPreviousHasOutputAdjustLayer(layer)
            if( ((CHIA==True) and (PHOA==False)) or
                ((CHIA==False) and (PHOA==True))):
                r = True
            elif(CHIA and PHOA and (len(layer['shape'])==4)):
                 r = True
            elif(CHIA and PHOA and (len(layer['shape'])==3)):
                 raise NotImplementedError("layer %s: don't know whether it was NHWC or not,"
                    " add the justfication here:\n%s"%(layer['name'], self))
        return r

    def nchw_ActionLayerNHWC(self, layer):
        self.convert_layer_to_nchw(layer)
        return False

    def get_consumers(self, layer, model=None):
        consumers = []
        if(model == None):
            model = self.lwnn_model
        for ly in model:
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

    def opt_IsLayerActivationAfter(self, layer, act, op):
        r = False
        if('inputs' in layer):
            inputs = self.get_layers(layer['inputs'])
            if((layer['op'] == act) and 
                   (len(inputs) == 1) and
                   (inputs[0]['op'] == op)):
                r = True
        return r

    def opt_MergeActivation(self, layer, activation):
        inputs = self.get_layers(layer['inputs'])
        inp = inputs[0]
        inp['activation'] = activation
        self.opt_RemoveLayer(layer)
        return True

    def opt_IsLayerReLUConv(self, layer):
        return self.opt_IsLayerActivationAfter(layer, 'Relu', 'Conv')

    def opt_MergeReLUConv(self, layer):
        return self.opt_MergeActivation(layer, 'Relu')

    def opt_IsLayerReLUDense(self, layer):
        return self.opt_IsLayerActivationAfter(layer, 'Relu', 'Dense')

    def opt_MergeReLUDense(self, layer):
        return self.opt_MergeActivation(layer, 'Relu')

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
        else:
            raise Exception("don't know how to fuse for %s shape %s"%(layer.name, c_w.shape))
        layer['weights'] = c_w
        layer['bias'] = c_b
        self.opt_RemoveLayer(bn)
        return True

    def opt_IsLayerReshapeBeforeSoftmax(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((layer['op'] == 'Reshape') and
               (len(consumers) == 1) and
               (consumers[0]['op'] == 'Softmax')):
            r = True
        return r

    def opt_IsLayerOutputWithOutput(self, layer):
        r = False
        if('inputs' in layer):
            inputs = self.get_layers(layer['inputs'])
            if((layer['op'] == 'Output') and
               (len(inputs) == 1) and
                (inputs[0]['op'] in ['Softmax', 'DetectionOutput'])):
                r = True
        return r

    def opt_RemoveOutputWithOutput(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        inp['Output'] = True
        self.opt_RemoveLayer(layer)
        return True

    def permute_shape(self, layer):
        if('permute' not in layer):
            shape = layer['shape']
            if(len(shape) == 3):
                layer['shape'] = [shape[0], shape[2], shape[1]]
        layer['permute'] = True

    def opt_PermuteReshapeSoftmax(self, layer):
        self.permute_shape(layer)
        softmax = self.get_consumers(layer)[0]
        self.permute_shape(softmax)
        return False

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
        for ly in [layer]+layers:
            self.opt_RemoveLayer(ly)
        return True

    def opt_IsLayerIdentity(self, layer):
        r = False
        if(layer['op'] == 'Identity'):
            r = True
        return r

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
        add['op'] = 'Dense'
        add['inputs'] = layer['inputs']
        add['weights'] = layer['weights']
        self.lwnn_model.remove(layer)
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

    def opt_IsLayerPooling1D(self, layer):
        r = False
        if((layer['op'] in ['MaxPool', 'AveragePool']) and 
            (len(layer['kernel_shape']) == 1)):
            r = True
        return r

    def opt_LayerPooling1D(self, layer):
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
        if('WeightsReordered' in layer):
            return False
        W = layer['weights']
        if(len(W.shape)==4):
            W = W.transpose(0,2,3,1)
        if(layer['group'] == 1):
            pass
        elif(layer['group'] == layer['shape'][1]):
            if(len(W.shape)==4):
                W = W.transpose(3,1,2,0)
        layer['weights'] = W
        layer['WeightsReordered'] = True
        return False

    def opt_IsLayerReshape(self, layer):
        r = False
        if(layer['op'] == 'Reshape'):
            r = True
        return r

    def opt_RemoveLayer(self, layer):
        consumers = self.get_consumers(layer)
        for ly in consumers:
            new_inputs = []
            inputs = self.get_layers(ly['inputs'])
            for inp in inputs:
                if(inp == layer):
                    new_inputs.append(self.get_layers(inp['inputs'])[0]['name'])
                else:
                    new_inputs.append(inp['name'])
            ly['inputs'] = new_inputs
        self.lwnn_model.remove(layer)
        return True

    def opt_IsLayerUnused(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((len(consumers) == 0) and
           ((layer['op'] != 'Output') and ('Output' not in layer))):
            r = True
        return r

    def opt_IsLayerTransposeCanBeRemoved(self, layer):
        r = False
        if((layer['op'] == 'Transpose') and
           (layer['perm'] == [0 , 2 , 3 , 1])):
            # LWNN is already NHWC
            r = True
        return r

    def opt_IsLayerConcatOnPriorBox(self, layer):
        r = False
        if(layer['op'] == 'Concat'):
            inputs = self.get_layers(layer['inputs'])
            r = True
            for inp in inputs:
                if(inp['op'] != 'PriorBox'):
                    r = False
        return r

    def opt_ReplaceAsConstant(self, layer):
        outputs = self.run()
        if(self.opt_IsLayerConcatOnPriorBox(layer)):
            layer['ConcatOnPriorBox'] = True
        const = outputs[layer['outputs'][0]]
        const = np.array(const, np.float32)
        layer['op'] = 'Const'
        layer['inputs'] = []
        layer['const'] = const
        return True

    def opt_IsLayerDetectionOutputWithConst(self, layer):
        r = False
        if(layer['op'] == 'DetectionOutput'):
            inputs = self.get_layers(layer['inputs'])
            if((len(inputs) == 3) and
               self.is_there_op(inputs, 'Const')):
                r = True
        return r

    def opt_MergeConstToDetectionOutput(self, layer):
        inputs = self.get_layers(layer['inputs'])
        const = None
        inputsL = []
        for inp in inputs:
            if(inp['op'] == 'Const'):
                const = inp
            else:
                inputsL.append(inp['name'])
        layer['priorbox'] = const['const']
        layer['inputs'] = inputsL
        return True

    def opt_LayerUnusedAction(self, layer):
        self.lwnn_model.remove(layer)
        return True

    def opt_IsTrainingOperators(self, layer):
        r = False
        if(layer['op'] in ['Dropout']):
            r = True
        return r

    def optimize(self, additions=[]):
        id = -1
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
                        id = -1
                        num_layers = len(self.lwnn_model)
                        break

    def toCstr(self, name):
        for s in ['/',':', '-']:
            name = name.replace(s, '_')
        return name

    def check(self):
        for id,layer in enumerate(self.lwnn_model):
            if('inputs' in layer):
                # check that inputs are before me
                inputs = self.get_layers(layer['inputs'],self.lwnn_model[:id])
                if(len(inputs) != len(layer['inputs'])):
                    raise Exception('layer %s inputs is not before me:\n%s'%(layer['name'], self))

    def prepare(self):
        # everthing is fine, fix name
        for layer in self.lwnn_model:
            layer['name'] = self.toCstr(layer['name'])
            if('inputs' in layer):
                layer['inputs'] = [self.toCstr(inp) for inp in layer['inputs']]
        self.is_model_channel_first()

    def __str__(self, model=None):
        if(model == None):
            model = self.lwnn_model
        cstr = 'LWNN Model %s:\n'%(self.name)
        order = ['name', 'op', 'shape','inputs', 'outputs', 'weights', 'bias', 'const']
        for layer in model:
            cstr += ' {'
            for k in order:
                if(k in layer):
                    v = layer[k]
                    if(k in ['weights','bias', 'const']):
                        cstr += '%s: %s, '%(k, v.shape)
                    else:
                        cstr += '%s: %s, '%(k,v)
            for k,v in layer.items():
                if(k not in order):
                    cstr += '%s: %s, '%(k,v)
            cstr += '}\n'
        return cstr
