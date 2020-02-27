# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from . import *
from .float import *
from .qformat import *
from lwnn2onnx import *
import pickle
import traceback

class LWNNUtil():
    def LN(self, name):
        '''return lwnn type layer name'''
        if(':' in name):
            name,_ = name.split(':')
        return name

    def get_layers(self, names, model=None):
        layers = []
        if(model == None):
            model = self.lwnn_model
        if(type(names) is str):
            for layer in model:
                if(layer['name'] == names):
                    return layer
            return None
        for name in names:
            for layer in model:
                if(layer['name'] == name):
                    layers.append(layer)
        return layers

    def get_consumers(self, layer, model=None):
        consumers = []
        if(model == None):
            model = self.lwnn_model
        for ly in model:
            if('inputs' not in ly): continue
            if(layer['name'] in ly['inputs']):
                consumers.append(ly)
        return consumers

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
           ((layer['op'] not in ['Output', 'Softmax', 'DetectionOutput', 'YoloOutput']) 
            and ('Output' not in layer))):
            r = True
        return r

    def opt_LayerUnusedAction(self, layer):
        self.lwnn_model.remove(layer)
        return True

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

    def optimize(self, additions=[]):
        id = -1
        num_layers = len(self.lwnn_model)
        while(id < (num_layers-1)):
            id += 1
            layer = self.lwnn_model[id]
            for isopt, optact, oname in self.OPTIMIER:
                if((((oname == None) and (len(additions) == 0)) 
                    or (oname in additions)) and isopt(layer)):
                    r = optact(layer)
                    if(True == r): # if there is remove action, restart optimization
                        id = -1
                        num_layers = len(self.lwnn_model)
                        break

    def toCstr(self, name):
        for s in ['/',':', '-', '.']:
            name = name.replace(s, '_')
        fc = name[0]
        if(fc.isdigit()):
            name = '_' + name
        return name

class LWNNLayer(dict):
    def __init__(self, **kwargs):
        try:
            super().__init__()
        except:
            dict.__init__(self)
        for k,v in kwargs.items():
            self[k] = v

    def __getattr__(self, key):
        return self[key]

    def __setattr__(self, key, v):
        self[key] = v

    def __str__(self):
        order = ['name', 'op', 'shape','inputs', 'outputs', 'weights','bias']
        def kv2s(k, v):
            cstr = ''
            try:
                cstr += '%s=t%s, '%(k, v.shape)
            except:
                if(k in ['top', 'topq']):
                    cstr += '%s=[ '%(k)
                    for top in v:
                        try:
                            cstr += '%s, '%(str(top.shape))
                        except:
                            cstr += '%s, '%(top)
                    cstr += '], '
                else:
                    if(k == 'name'):
                        cstr += '%s, '%(v)
                    else:
                        cstr += '%s=%s, '%(k,v)
            return cstr
        cstr = 'LWNNLayer('
        for k in order:
            if(k in self):
                cstr += kv2s(k, self[k])
        for k,v in self.items():
            if(k not in order):
                cstr += kv2s(k, v)
        cstr = cstr[:-2] + ')'
        return cstr

class LWNNModel(LWNNUtil):
    def __init__(self, converter, name, **kwargs):
        self.OPTIMIER = [
            (self.nchw_IsLayerNHWC, self.nchw_ActionLayerNHWC, None),
            (self.nchw_IsInputAdjustLayer, self.nchw_ActionInputAdjustLayer, None),
            (self.nchw_IsOutputAdjustLayer, self.opt_RemoveLayer, None),
            (self.opt_IsLayerUnused, self.opt_LayerUnusedAction, None),
            (self.opt_IsLayerFakeQuantize, self.opt_LayerFakeQuantize, None),
            (self.opt_IsLayerHasInitializer, self.opt_LayerHasInitializer, None),
            (self.opt_IsLayerDense, self.opt_LayerDense, None),
            (self.opt_IsLayerConv1D, self.opt_LayerConv1D, None),
            (self.opt_IsLayerPooling1D, self.opt_LayerPooling1D, None),
            (self.opt_IsLayerConvBeforeBN, self.opt_FuseConvBN, None),
            (self.opt_IsLayerConv, self.opt_LayerConvWeightsReorder, None),
            (self.opt_IsLayerConvTranspose, self.opt_LayerConvTransposeWeightsReorder, None),
            (self.opt_IsTrainingOperators, self.opt_RemoveLayer, None),
            (self.opt_IsLayerConcatWithOneOnly, self.opt_LayerConcatWithOneOnly, None),
            (self.opt_IsLayerConcatOnPriorBox, self.opt_LayerConcatOnPriorBox, None),
            (self.opt_IsLayerDetectionOutputWithConst, self.opt_MergeConstToDetectionOutput, None),
            (self.opt_IsLayerReshapeBeforeSoftmax, self.opt_PermuteReshapeSoftmax, None),
            (self.opt_IsLayerOutputWithoutConsumers, self.opt_LayerOutputWithoutConsumers, None),
            (self.opt_IsLayerOutputWithOutput, self.opt_RemoveOutputWithOutput, None),
            (self.opt_IsLayerClipRelu, self.opt_LayerClip2Relu, None),
            (self.opt_IsLayerFlatten, self.opt_LayerFlatten2Reshape, None),
            (self.opt_IsLayerPad, self.opt_LayerPad, None),
            (self.opt_IsLayerTransposeCanBeRemoved, self.opt_RemoveLayer, 'RemoveTranspose'),
            (self.opt_IsLayerIdentity, self.opt_RemoveLayer, 'RemoveIdentity'),
            (self.opt_IsLayerReshape, self.opt_RemoveLayer, 'RemoveReshape'),
            (self.opt_IsLayerReLUConv, self.opt_MergeReLUConv, 'MergeReLUConv'),
            (self.opt_IsLayerReLUDense, self.opt_MergeReLUDense, 'MergeReLUDense'),
            ]
        self.is_model_channel_first_cached=None
        self.converter = converter
        self.name = self.toCstr(name)
        self.converter.save(self.path)
        self.lwnn_model = self.converter.convert()
        # optimization and convert to NCHW if origin model is NHWC
        self.prepare()
        self.omodel = self.clone()
        if(not (('notRmIdentity' in kwargs) and (kwargs['notRmIdentity']==True))):
            self.optimize(['RemoveIdentity'])
        self.omodel = self.clone()
        self.optimize()
        self.omodel = self.clone()
        self.save()
        self.optimize(['RemoveTranspose'])
        self.check()
        print(self)

    def save(self):
        try:
            lwnn2onnx(self.omodel, '%s.lwnn.onnx'%(self.path))
        except:
            traceback.print_exc()
        try:
            pickle.dump(self.lwnn_model, open('%s.pkl'%(self.path), 'wb'), True)
        except Exception as e:
            print(e)

    @property
    def input(self):
        return self.converter.input
    @property
    def output(self):
        return self.converter.output

    def run(self, feed=None, model=None):
        if(model == None):
            model = self.omodel
        O = self.converter.run(feed, model=model)
        outputs = {}
        for k,v in O.items():
            outputs[self.toCstr(k)] = v
        return outputs

    def clone_layer(self, layer):
        L = LWNNLayer()
        for k,v in layer.items():
            if(type(v) in [list, tuple]):
                L[k] = list(v)
            else:
                L[k] = v
        return L

    def clone(self):
        model = []
        for ly in self.lwnn_model:
            model.append(self.clone_layer(ly))
        return model

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
        elif(layer['op'] not in ['Conv', 'MaxPool', 'AveragePool', 'Upsample']):
            CHIA = self.nchw_IsConsumerHasInputAdjustLayer(layer)
            PHOA = self.nchw_IsPreviousHasOutputAdjustLayer(layer)
            if( ((CHIA==True) and (PHOA==False)) or
                ((CHIA==False) and (PHOA==True))):
                r = True
            elif(CHIA and PHOA and (len(layer['shape'])==4)):
                r = True
            elif(CHIA and PHOA and (len(layer['shape'])==3)):
                if(layer['op'] in ['Clip','Relu', 'Output']):
                    r = True
                else:
                    raise NotImplementedError("layer %s: don't know whether it was NHWC or not,"
                    " add the justfication here:\n%s"%(layer['name'], self))
        return r

    def nchw_ActionLayerNHWC(self, layer):
        self.convert_layer_to_nchw(layer)
        return False

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
            raise Exception("don't know how to fuse for %s shape %s"%(layer['name'], c_w.shape))
        layer['weights'] = c_w
        layer['bias'] = c_b
        self.opt_RemoveLayer(bn)
        return True

    def opt_IsLayerReshapeBeforeSoftmax(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((layer['op'] in ['Reshape', 'Flatten']) and
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
               (consumers[0]['op'] == 'Add') and
               ('bias' in consumers[0])):
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
        if('dilations' in layer):
            dilation = layer['dilations'][0]
            layer['dilations'] = [dilation, dilation]
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

    def opt_IsLayerConvTranspose(self, layer):
        r = False
        if(layer['op'] == 'ConvTranspose'):
            r = True
        return r

    def opt_LayerConvTransposeWeightsReorder(self, layer):
        # DeConv:  (C x M/group x kH x kW), -> [M/group x kH x kW x C]
        if('WeightsReordered' in layer):
            return False
        W = layer['weights']
        if(len(W.shape)==4):
            W = W.transpose(1,2,3,0)
            W = np.rot90(W, k=2, axes=(1, 2))
        layer['weights'] = W
        layer['WeightsReordered'] = True
        return False

    def opt_IsLayerReshape(self, layer):
        r = False
        if(layer['op'] == 'Reshape'):
            r = True
        return r

    def opt_IsLayerClipRelu(self, layer):
        r = False
        if((layer['op'] == 'Clip') and
           (('min' not in layer) or (layer['min']==0.0))):
            r = True
        return r

    def opt_LayerClip2Relu(self, layer):
        layer['op'] = 'Relu'
        return False

    def opt_IsLayerOutputWithoutConsumers(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((layer['op'] in ['Softmax', 'DetectionOutput', 'YoloOutput']) and
           (len(consumers) == 0)):
            r = True
        return r

    def opt_LayerOutputWithoutConsumers(self, layer):
        layer['Output'] = True
        return False

    def opt_IsLayerFlatten(self, layer):
        r = False
        if(layer['op'] == 'Flatten'):
            r = True
        return r

    def opt_LayerFlatten2Reshape(self, layer):
        layer['op'] = 'Reshape'
        return False

    def opt_IsLayerPad(self, layer):
        r = False
        if(layer['op'] == 'Pad'):
            r = True
        return r

    def opt_LayerPad(self, layer):
        if('pads' not in layer):
            padsL = []
            padsR = []
            inputs = self.get_layers(layer['inputs'])
            ishape = inputs[0]['shape']
            oshape = layer['shape']
            for di, do in zip(ishape, oshape):
                padding = int((do-di)/2)
                padsL.append(padding)
                padsR.append(padding)
            layer['pads'] = padsL + padsR
        return False

    def opt_IsLayerFakeQuantize(self, layer):
        r = False
        if(layer['op'] == 'FakeQuantize'):
            r = True
        return r

    def opt_IsLayerHasInitializer(self, layer):
        r = False
        if((layer['op'] == 'Conv') and (len(layer['inputs']) > 1)):
            r = True
        return r

    def opt_IsLayerTransposeCanBeRemoved(self, layer):
        r = False
        if((layer['op'] == 'Transpose') and
           ( (list(layer['perm']) == [0 , 2 , 3 , 1]) or 
             (list(layer['perm']) == [0 , 2 , 1]) )):
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

    def create_priorbox_model(self, layer):
        model = []
        for ly in self.get_layers(layer['inputs']):
            L = self.clone_layer(ly)
            inputs = self.get_layers(L['inputs'])
            for inp in inputs:
                if(inp['op'] in ['Input', 'Unsqueeze']):
                    L['image_shape'] = inp['shape']
                else:
                    L['feature_shape'] = inp['shape']
            L['inputs'] = []
            model.append(L)
        model.append(self.clone_layer(layer))
        return model

    def opt_LayerConcatOnPriorBox(self, layer):
        outputs = self.run(model=self.create_priorbox_model(layer))
        oname = layer['outputs'][0]
        const = outputs[oname]
        const = np.array(const, np.float32)
        layer['ConcatOnPriorBox'] = True
        layer['op'] = 'Constant'
        layer['inputs'] = []
        layer['const'] = const 
        return True

    def opt_LayerConcatWithOneOnly(self, layer):
        inp = self.get_layers(layer['inputs'])[0]
        if(inp['op'] == 'Split'):
            self.opt_RemoveLayer(inp)
        return self.opt_RemoveLayer(layer)

    def opt_IsLayerConcatWithOneOnly(self, layer):
        r = False
        if(layer['op'] == 'Concat'):
            inputs = self.get_layers(layer['inputs'])
            if(len(inputs) == 1):
                r = True
        return r

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

    def opt_LayerFakeQuantize(self, layer):
        r = self.opt_RemoveLayer(layer)
        return r

    def opt_LayerHasInitializer(self, layer):
        op = layer['op']
        inputs = self.get_layers(layer['inputs'])
        if(op == 'Conv'):
            layer['inputs'] = [inputs[0]['name']]
            layer['weights'] = inputs[1]['const']
            if(len(inputs) == 3):
                layer['bias'] = inputs[2]['const']
            else:
                M = layer['weights'].shape[0]
                layer['bias'] = np.zeros((M), np.float32)
        return True

    def opt_IsTrainingOperators(self, layer):
        r = False
        if(layer['op'] in ['Dropout']):
            r = True
        return r

    def check(self):
        for id,layer in enumerate(self.lwnn_model):
            if('inputs' in layer):
                # check that inputs are before me
                LI = layer['inputs']
                inputs = self.get_layers(LI,self.lwnn_model[:id])
                eLI = []
                for inp in LI:
                    if(inp not in eLI):
                        eLI.append(inp)
                if(len(eLI) != len(inputs)):
                    raise Exception('layer %s inputs is not before me:\n%s'%(layer['name'], self))

    def prepare(self):
        # everthing is fine, fix name
        for layer in self.lwnn_model:
            layer['name'] = self.toCstr(layer['name'])
            if('inputs' in layer):
                layer['inputs'] = [self.toCstr(inp) for inp in layer['inputs']]
            layer['outputs'] = [self.toCstr(out) for out in layer['outputs']]
        self.is_model_channel_first()

    def __str__(self, model=None):
        if(model == None):
            model = self.lwnn_model
        cstr = 'LWNN Model %s:\n'%(self.name)
        for layer in model:
            cstr += ' %s\n'%(layer)
        return cstr
