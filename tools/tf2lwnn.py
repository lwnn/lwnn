# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>

from lwnn.core import *
from lwnn.tfseqs import *
import tensorflow as tf
import numpy as np
import os
import onnx
from onnx.shape_inference import infer_shapes
try:
    import tf2onnx
    from tf2onnx.tfonnx import process_tf_graph, tf_optimize
    from onnx2lwnn import OnnxConverter
except Exception as e:
    print('Warning:', e)

try:
    TF_VERSION = eval(str(tf.VERSION).replace('.', ','))
except:
    TF_VERSION = eval(str(tf.__version__).replace('.', ','))

IS_TF_V2 = False
if(TF_VERSION[0] == 2):
    IS_TF_V2 = True
    tfFastGFile = tf.io.gfile.GFile
    tfGraphDef = tf.compat.v1.GraphDef
    tfSession = tf.compat.v1.Session
    raise NotImplementedError('tf version 2 is not supported for now')
else:
    tfFastGFile = tf.gfile.FastGFile
    tfGraphDef = tf.GraphDef
    tfSession = tf.Session

__all__ = ['tf2lwnn', 'TfConverter']

class TfConverter(LWNNUtil):
    def __init__(self, graph_def, name, **kwargs):
        if('output_node' in kwargs):
            LWNNOutputNodes.remove('Softmax')
        self.OPTIMIER = [
            (self.opt_IsLayerOpInTranslator, self.opt_LayerOpInTranslator, None),
            (self.opt_IsLayerConvBeforeBiasAdd, self.opt_FuseConvBiasAdd, None),
            (self.opt_IsLayerLSTM, self.opt_LayerLSTM, None),
            (self.opt_IsLayerMfcc, self.opt_LayerMfcc, None),
            (self.opt_IsLayerSoftmax, self.opt_LayerSoftmax, 'graph_match'),
            (self.opt_IsLayerNorm, self.opt_LayerNorm, 'graph_match'),
            (self.opt_IsLayerEinsum1, self.opt_LayerEinsum1, 'graph_match'),
            (self.opt_IsLayerEinsum2, self.opt_LayerEinsum2, 'graph_match'),
            (self.opt_IsLayerClip, self.opt_LayerClip, 'graph_match'),
            (self.opt_IsLayerConstSlice, self.opt_LayerConstSlice, 'graph_match'),
            (self.opt_IsLayerConstFill, self.opt_LayerConstFill, 'graph_match'),
            (self.opt_IsLayerConstMul, self.opt_LayerConstMul, 'graph_match'),
            (self.opt_IsLayerReshapeAfterReshape, self.opt_LayerReshapeAfterReshape, None),
            (self.opt_IsLayerReshapeNotNecesary, self.opt_RemoveLayer, None),
            #(self.opt_IsLayerUnused, self.opt_LayerUnusedAction, None),
            ]
        self.TRANSLATOR = {
            'Reshape': self.to_LayerReshape,
            'DecodeWav': self.to_LayerDecodeWav,
            'MatMul': self.to_LayerMatMul,
            'Add': self.to_LayerAdd,
            'Constant': self.to_LayerConst,
            'BlockLSTM': self.to_LayerBlockLSTM,
            'Conv': self.to_LayerConv,
            'BatchNormalization': self.to_LayeBatchNormalization,
            'Concat': self.to_LayeConcat,
            'AveragePool': self.to_LayerPool,
            'MaxPool': self.to_LayerPool,
            'Transpose': self.to_LayerTranspose }
        self.opMap = {
            'Placeholder': 'Input',
            'Const': 'Constant',
            'BiasAdd': 'Add',
            'ExpandDims': 'Reshape',
            'Squeeze': 'Reshape',
            'Conv2D': 'Conv',
            'Minimum': 'Min',
            'FusedBatchNorm': 'BatchNormalization',
            'ConcatV2': 'Concat',
            'ResizeBilinear': 'Resize',
            'AvgPool': 'AveragePool',
            'AddV2': 'Add',
            'GatherV2': 'Gather',
            'BatchMatMulV2': 'BatchMatMul',
            }
        if(type(graph_def) == str):
            with tfFastGFile(graph_def, 'rb') as f:
                graph_def = tfGraphDef()
                graph_def.ParseFromString(f.read())
        self.graph_def = graph_def
        self.name = name
        self.sess = tfSession()
        _ = tf.import_graph_def(self.graph_def, name=self.name)
        if(IS_TF_V2):
            pass #tf.summary.create_file_writer('./graphs')
        else:
            #tf.summary.FileWriter('./graphs', self.sess.graph)
            self.sess.run(tf.global_variables_initializer())
        self.tensors = {}
        for node in self.graph_def.node:
            if(node.op in ['Assert']): continue
            x = self.sess.graph.get_tensor_by_name('%s/%s:0'%(self.name, node.name))
            self.tensors[self.c_str(node.name)] = x
        self.kwargs = kwargs
        if('dynamic_shape' in self.kwargs):
            self.dynamic_shape = self.kwargs
        else:
            self.dynamic_shape = False
        self.convert()

    @property
    def model(self):
        return self.lwnn_modelo

    def eval(self, layer, feed=None):
        if(type(layer) == str):
            lname = layer
        else:
            lname = layer.name
        if(feed is not None):
            rs = self.sess.run(self.tensors[self.c_str(lname)], feed)
        else:
            rs = self.sess.run(self.tensors[self.c_str(lname)])
        if(type(rs) == np.float32):
            rs = float(rs)
        elif(type(rs) == np.int32):
            rs = int(rs)
        return rs

    def get_tensor(self, name):
        return self.tensors[self.c_str(name)]

    def get_layers_by_scope(self, scope):
        L=[]
        for l in self.lwnn_model:
            if(l.name.startswith(scope)):
                L.append(l)
        return L

    def save(self, path):
        try:
            lwnn2onnx(self.lwnn_model, '%s.tf.onnx'%(path))
        except:
            traceback.print_exc()

    def has_field(self, attr, field):
        try:
            return attr.HasField(field)
        except ValueError:
            try:
                _ = attr.__getattribute__(field)
                return True
            except ValueError:
                return False

    def to_LayerCommon(self, node):
        op = node.op
        if(op in self.opMap):
            op = self.opMap[op]
        layer = LWNNLayer(name=node.name, op=op)
        if(self.has_field(node, 'input')):
            inputs = []
            input_ref=[]
            for inp in node.input:
                try:
                    name,id= inp.split(':')
                    inputs.append(name.replace('^', ''))
                    input_ref.append(eval(id))
                except ValueError:
                    inputs.append(inp.replace('^', ''))
                    input_ref.append(0)
            if(len(inputs) > 0):
                layer['inputs'] = inputs
                layer['input_ref'] = input_ref
        for k in node.attr:
            attr = node.attr[k]
            if(self.has_field(attr,'type')):
                if('DT_STRING' in str(attr)):
                    attr = 'string'
                elif('DT_INT32' in str(attr)):
                    attr = 'int32'
                elif('DT_INT64' in str(attr)):
                    attr = 'int64'
                elif('DT_FLOAT' in str(attr)):
                    attr = 'float'
                elif('DT_BOOL' in str(attr)):
                    attr = 'bool'
                elif('DT_RESOURCE' in str(attr)):
                    attr = 'resource' 
                else:
                    raise NotImplementedError('type %s of node %s is not supported'%(attr, node))
            elif(self.has_field(attr,'shape')):
                attr = [dim.size for dim in attr.shape.dim]
            elif(self.has_field(attr,'i')):
                attr = attr.i
            elif(self.has_field(attr,'b')):
                attr = attr.b
            elif(self.has_field(attr,'f')):
                attr = attr.f
            elif(self.has_field(attr,'s')):
                attr = attr.s.decode('utf-8')
            elif(self.has_field(attr,'list')):
                L = attr.list
                if(self.has_field(L,'i')):
                    attr = [i for i in L.i]
                elif(self.has_field(L,'s')):
                    attr = [s.decode('utf-8') for s in L.s]
                else:
                    raise
            elif(self.has_field(attr,'tensor')):
                tensor = attr.tensor
                shape = [dim.size for dim in tensor.tensor_shape.dim]
                if('DT_INT32' in str(tensor)):
                    dtype = np.int32
                elif('DT_INT64' in str(tensor)):
                    dtype = np.int64
                elif('DT_FLOAT' in str(tensor)):
                    dtype = np.float32 
                elif('DT_STRING' in str(tensor)):
                    dtype = 'string' 
                else:
                    raise NotImplementedError('type of tensor %s is not supported'%(attr))
                try:
                    attr = np.copy(np.ndarray(
                            shape=shape,
                            dtype=dtype,
                            buffer=tensor.tensor_content))
                except:
                    if(dtype == np.int32):
                        attr = np.asarray(tensor.int_val, dtype=dtype)
                    elif(dtype == np.float32):
                        attr = np.asarray(tensor.float_val, dtype=dtype)
                    elif(dtype == 'string'):
                        attr = np.copy(np.ndarray(
                            shape=(len(tensor.string_val[0])),
                            dtype=np.int8,
                            buffer=tensor.string_val[0]))
                    else:
                        raise
            else:
                raise NotImplementedError('attr %s=%s of node %s is not supported'%(k, attr, node))
            layer[k] = attr
        tensor = self.get_tensor(node.name)
        try:
            shape = tensor.shape.as_list()
        except ValueError:
            if(('shapes' in self.kwargs) and (node.name in self.kwargs['shapes'])):
                shape = self.kwargs['shapes'][node.name]
                if(type(shape) == int):
                    shape = [shape]
                else:
                    shape = list(shape)
            else:
                shape = []
        layer['outputs'] = [node.name]
        if(len(shape) > 0):
            for i, s in enumerate(shape):
                if(s in [None, 0]):
                    shape[i] = -1 if self.dynamic_shape else 1
            shape[0] = 1 # 1 batch mode for lwnn
            layer['shape'] = shape
        elif(('dtype' in layer) and (layer.dtype == 'bool')):
            layer.shape = [1]
        return layer

    def to_LayerReshape(self, layer):
        if('shape' not in layer):
            _, shape = self.get_layers(layer.inputs)
            layer.shape = self.eval(shape).tolist()
        layer.inputs = layer.inputs[:1]
        inp = self.get_layers(layer.inputs[0])
        if((-1 in layer.shape) and (not self.dynamic_shape)):
            dims = 1
            for s in inp.shape:
                dims = dims*s
            for i,s in enumerate(layer.shape):
                if(s != -1):
                    dims = int(dims/s)
                else:
                    axis = i
            layer.shape[axis] = dims

    def to_LayerDecodeWav(self, layer):
        layer.outputs.append('%s:1'%(layer.name))

    def to_LayerMatMul(self, layer):
        _, weights = self.get_layers(layer.inputs)
        layer.weights = self.eval(weights)
        if(layer.transpose_b):
            layer.weights = layer.weights.transpose()
        layer.inputs = layer.inputs[:1]

    def to_LayerBlockLSTM(self, layer):
        inputs = self.get_layers(layer.inputs)
        x,w,b = inputs[1],inputs[4],inputs[-1]
        W = self.eval(w)
        B = self.eval(b)
        W = W.transpose(1,0)
        layer.input_size = I = x.shape[-1]
        layer.hidden_size = H = int(B.shape[0]/4)
        O = W.shape[-1]-I   # output size
        W,R = W[:,:I], W[:, I:]
        Wi,Wc,Wf,Wo = W.reshape(4,-1,I)
        Ri,Rc,Rf,Ro = R.reshape(4,-1,O)
        Wbi,Wbc,Wbf,Wbo = B.reshape(4, H)
        Rbi,Rbc,Rbf,Rbo = np.zeros((4, H), np.float32)
        # ONNX W,R,B, not bidirectional
        W = np.concatenate([Wi, Wo, Wf, Wc], axis=0).reshape(1, 4*H, I)
        R = np.concatenate([Ri, Ro, Rf, Rc], axis=0).reshape(1, 4*H, O)
        B = np.concatenate([Wbi, Wbo, Wbf, Wbc, Rbi, Rbo, Rbf, Rbc], axis=0).reshape(1, 8*H)
        layer.op = 'LSTM'
        layer.W = W
        layer.R = R
        layer.B = B
        layer.inputs = [x.name]
        for c in self.get_consumers(layer):
            if(c.op == 'Mul'):
                mul = c
                break
        for c in self.get_consumers(mul):
            if(c.op == 'Reshape'):
                o = c
                break
        o.inputs = [layer.name]
        return False

    def to_LayerConv(self, layer):
        inputs = self.get_layers(layer.inputs)
        W = self.eval(inputs[1])
        if(len(inputs) > 2):
            layer.bias = self.eval(inputs[2])
        else:
            C = W.shape[-1]
            B = np.zeros((C), np.float32)
        W = W.transpose(3,2,0,1)
        if('group' not in layer):
            layer.group = 1
        if(('strides' not in layer) or (len(layer.strides) == 0)):
            layer.strides = [1, 1]
        else:
            layer.strides = layer.strides[1:3]
        if(('dilations' not in layer) or (len(layer.dilations) == 0)):
            layer.dilations = [1, 1]
        else:
            layer.dilations = layer.dilations[1:3]
        layer.kernel_shape = W.shape[2:]
        self.infer_conv_or_pool_shape_and_padding(layer)
        layer.weights = W
        layer.bias = B
        layer.inputs = layer.inputs[:1]

    def to_LayerPool(self, layer):
        layer.strides = layer.strides[1:3]
        layer.kernel_shape = layer.ksize[1:3]
        self.infer_conv_or_pool_shape_and_padding(layer)

    def to_LayeBatchNormalization(self, layer):
        x,scale,bias,mean,var = self.get_layers(layer.inputs)
        if(x.op == 'Switch'):
            #L = self.get_layers_by_scope(layer.name.split('cond')[0])
            #self.graph_helper(L)
            for merge in self.lwnn_model:
                if(merge.op == 'Merge'):
                    if(self.graph_match(merge, TFSEQ_BATCHNORM_COND)):
                        graph = self.get_matched_graph()
                        layer.scale = self.eval(graph[44])
                        layer.bias = self.eval(graph[46])
                        layer.var = self.eval(graph[40])
                        layer.mean = self.eval(graph[42])
                        layer.inputs = [graph[48].name, graph[47].name]
                        layer.name = merge.name
                        assert(graph[33] == layer)
            assert(0)
        else:
            layer.scale = self.eval(scale)
            layer.bias = self.eval(bias)
            layer.var = self.eval(var)
            layer.mean = self.eval(mean)
            layer.inputs = layer.inputs[:1]

    def to_LayeConcat(self, layer):
        N = layer.N
        axis = self.get_layers(layer.inputs[N])
        try:
            layer.axis = self.eval(axis)
        except:
            layer.axis = axis.value[0]
        layer.inputs = layer.inputs[:N]

    def to_LayerAdd(self, layer):
        _, bias = self.get_layers(layer.inputs)
        try:
            if(bias.op == 'Constant'):
                layer.bias = bias.value
            else:
                layer.bias = self.eval(bias)
            layer.inputs = layer.inputs[:1]
            return True
        except Exception as e:
            pass #print('WARNING: %s\n\tBias: %s\n\t%s: %s'%(layer, bias, type(e), e))

    def to_LayerConst(self, layer):
        layer.const = layer.value
        layer.shape = layer.value.shape
        if('inputs' in layer):
            del layer['inputs']

    def to_LayerTranspose(self, layer):
        _, perm = self.get_layers(layer.inputs)
        layer.perm = self.eval(perm).tolist()
        layer.inputs = layer.inputs[:1]

    def opt_IsLayerOpInTranslator(self, layer):
        r = False
        if((layer.op in self.TRANSLATOR) and ('translated' not in layer)):
            r = True
        return r

    def opt_LayerOpInTranslator(self, layer):
        layer['translated'] = True
        return self.TRANSLATOR[layer.op](layer)

    def opt_IsLayerConvBeforeBiasAdd(self, layer):
        r = False
        consumers = self.get_consumers(layer)
        if((layer['op'] == 'Conv') and
               (len(consumers) == 1) and
               (consumers[0]['op'] == 'Add') and
               ('bias' in consumers[0])):
            r = True
        return r

    def opt_FuseConvBiasAdd(self, layer):
        biasAdd = self.get_consumers(layer)[0]
        layer.bias = biasAdd.bias
        for n,v in layer.items():
            if(n not in ['name', 'inputs', 'outputs']):
                biasAdd[n] = v
        return self.opt_RemoveLayer(layer)

    def opt_IsLayerLSTM(self, layer):
        r = False
        if(layer.op == 'MatMul'):
            for n in ['basic_lstm_cell','lstm_cell']:
                if(n in layer.name):
                    layer.lstm_type = n
                    r = True
                    break
        return r

    def getLSTMLayers(self, layer):
        # my way to handle LSTM wildly,
        # tf2onnx is much more precisely with a graph re match,
        # but too complicated and I find that it didn't works good
        scope = layer.name.split(layer.lstm_type)[0]
        if('while' in scope):
            scope = scope.split('while')[0]
        L=[]
        K = {}
        for l in self.lwnn_model:
            if(l.name.startswith(scope)):
                L.append(l)
                for key, condition in [
                    ('b',lambda l: (l.op == 'Add') and ('bias' in l) and (list(l.bias.shape) != [1])),
                    ('pi', lambda l: (l.op == 'Enter') and ('w_i_diag/read' in l.inputs[0])),
                    ('pf', lambda l: (l.op == 'Enter') and ('w_f_diag/read' in l.inputs[0])),
                    ('po', lambda l: (l.op == 'Enter') and ('w_o_diag/read' in l.inputs[0])),
                    ('pj', lambda l: (l.op == 'Enter') and ('projection/kernel/read' in l.inputs[0])),
                    ('o', lambda l: l.op == 'Exit'),
                    ('i', lambda l: l.op == 'TensorArrayScatterV3')]:
                    if(condition(l)):
                        if(key not in K):
                            K[key] = l
                        else:
                            raise Exception('The scope %s has too much %s op: %s'%(scope, l.op, l))
        return L, K

    def handleTranslatorOnce(self, op):
        L = []
        for l in self.lwnn_model:
            if(l.op == op):
                L.append(l)
        for l in L:
            self.opt_LayerOpInTranslator(l)

    def opt_LayerLSTM(self, layer):
        self.handleTranslatorOnce('Add') # depends on op Add translate firstly
        L, K = self.getLSTMLayers(layer)
        i,b,o=K['i'],K['b'],K['o']
        inp = self.get_layers(i.inputs)[2]
        W = layer.weights.transpose(1,0)
        B = b.bias
        layer.input_size = I = inp.shape[-1]
        layer.hidden_size = H = int(B.shape[0]/4)
        O = W.shape[-1]-I   # output size
        W,R = W[:,:I], W[:, I:]
        Wi,Wc,Wf,Wo = W.reshape(4,-1,I)
        Ri,Rc,Rf,Ro = R.reshape(4,-1,O)
        Wbi,Wbc,Wbf,Wbo = B.reshape(4, H)
        Rbi,Rbc,Rbf,Rbo = np.zeros((4, H), np.float32)
        # ONNX W,R,B, not bidirectional
        W = np.concatenate([Wi, Wo, Wf, Wc], axis=0).reshape(1, 4*H, I)
        R = np.concatenate([Ri, Ro, Rf, Rc], axis=0).reshape(1, 4*H, O)
        # Wbf+1 is experience learned
        B = np.concatenate([Wbi, Wbo, Wbf+1, Wbc, Rbi, Rbo, Rbf, Rbc], axis=0).reshape(1, 8*H)
        layer.op = 'LSTM'
        layer.W = W
        layer.R = R
        layer.B = B
        if('pi' in K):
            pi,pf,po = K['pi'],K['pf'],K['po']
            pi = self.eval(pi)
            pf = self.eval(pf)
            po = self.eval(po)
            layer.P = np.concatenate([pi,pf,po]).reshape(1, -1)
        if('pj' in K):
            layer.PRJECTION = self.eval(K['pj']).transpose(1,0).reshape(1, -1)
        del layer['weights']
        if(inp.op == 'Transpose'):
            layer.inputs = inp.inputs
        else:
            layer.inputs = [inp.name]
        layer.outputs = [o.name]
        # infer LSTM output shape
        itensor = self.get_tensor(inp.name)
        otensor = self.get_tensor(o.name)
        feeds = np.random.uniform(low=-1, high=1, size=itensor.shape)
        lstm_o = self.sess.run(otensor, {itensor:feeds})
        layer.shape = lstm_o.shape # shape to be used to decice output Y or Y_h
        layer.name = o.name

    def opt_IsLayerMfcc(self, layer):
        r = False
        if((layer.op == 'Mfcc') and (len(layer.inputs) > 1)):
            r = True
        return r

    def opt_LayerMfcc(self, layer):
        spectrogram, decode = self.get_layers(layer.inputs)
        layer.magnitude_squared = spectrogram.magnitude_squared
        layer.window_size = spectrogram.window_size
        layer.stride = spectrogram.stride
        if(decode.op == 'DecodeWav'):
            layer.desired_samples = decode.desired_samples
            layer.desired_channels = decode.desired_channels
            layer.inputs = decode.inputs
        else:
            layer.desired_samples = self.eval(decode)
            layer.desired_channels = 1
            layer.inputs = spectrogram.inputs

    def opt_IsLayerSoftmax(self, layer):
        graph = { 'Sequence': {0:'RealDiv', 1:'Exp', 2:'Sum', 3:'?',
                               4:'Sub', 5:'?', 6:'Max', 7:'?'},
                  'Connection': {0:[1,2], 1:[4], 2:[1,3], 3:[], 4:[5,6], 5:[], 6:[5,7], 7:[]}
                }
        return self.graph_match(layer, graph)

    def opt_LayerSoftmax(self, layer):
        graph = self.get_matched_graph()
        inp = graph[5]
        axis = self.eval(graph[7])
        layer.op = 'Softmax'
        layer.axis = axis
        layer.inputs = [inp.name]

    def opt_IsLayerNorm(self, layer):
#         if(('LayerNorm' in layer.name) and (layer.op == 'Add')):
#             scope = layer.name.split('batchnorm')[0]
#             L = self.get_layers_by_scope(scope)
#             self.graph_helper(L)
        return self.graph_match(layer, TFSEQ_LAYER_NORM)

    def opt_LayerNorm(self, layer):
        graph = self.get_matched_graph()
        layer.op = 'LayerNorm'
        x = graph[18]
        layer.gamma = self.eval(graph[15])
        layer.beta = self.eval(graph[17])
        layer.inputs = [x.name]

    def opt_IsLayerEinsum1(self, layer):
#         scope = 'bert/encoder/embedding_hidden_mapping_in/einsum'
#         if(layer.name == 'bert/encoder/embedding_hidden_mapping_in/add'):
#             L = self.get_layers_by_scope(scope)
#             L.append(layer)
#             self.graph_helper(L)
        return self.graph_match(layer, TFSEQ_EINSUM1)

    def opt_LayerEinsum1(self, layer):
        graph = self.get_matched_graph()
        layer.op = 'Dense'
        x = graph[22]
        reshape1 = graph[5]
        matmul = graph[4]
        reshape2 = graph[1]
        W = self.eval(graph[21])
        B = self.eval(graph[20])
        matmul.weights = W
        matmul.bias = B
        matmul.op = 'Dense'
        reshape1.inputs = [x.name]
        matmul.inputs = [reshape1.name]
        layer.op = reshape2.op
        layer.inputs = reshape2.inputs

    def opt_IsLayerEinsum2(self, layer):
#         scope = 'bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/einsum'
#         if(layer.name == 'bert/encoder/transformer/group_0/layer_0/inner_group_0/attention_1/self/query/add'):
#             L = self.get_layers_by_scope(scope)
#             L.append(layer)
#             self.graph_helper(L)
        return self.graph_match(layer, TFSEQ_EINSUM2)

    def opt_LayerEinsum2(self, layer):
        graph = self.get_matched_graph()
        x = graph[28]
        reshape1 = graph[10]
        matmul = graph[7]
        reshape2 = graph[3]
        transpose = graph[1]
        W = self.eval(graph[8])
        B = self.eval(graph[27])
        B = B.transpose().reshape(-1)
        matmul.weights = W
        matmul.bias = B
        matmul.op = 'Dense'
        reshape1.inputs = [x.name]
        matmul.inputs = [reshape1.name]
        reshape2.inputs = [matmul.name]
        layer.op = transpose.op
        layer.inputs = transpose.inputs

    def opt_IsLayerClip(self, layer):
        graph = { 'Sequence': {0:'Neg', 1:'Relu', 2:'Neg', 3:'?'},
                  'Connection': {0:[1], 1:[2], 2:[3], 3:[]}
                }
        return self.graph_match(layer, graph)

    def opt_LayerClip(self, layer):
        graph = self.get_matched_graph()
        inp = graph[3]
        layer.op = 'Clip'
        layer.max = 0
        layer.min = -np.inf
        layer.inputs = [inp.name]

    def opt_IsLayerConstSlice(self, layer):
        graph = { 'Sequence': {0:'Slice', 1:'Constant', 2:'Constant',
                               3:'Pack', 4:'StridedSlice', 5:'Constant',
                               6:'Shape', 7:'Constant', 8:'Constant', 9:'Constant'},
                  'Connection': {0:[1,2,3], 1:[], 2:[], 3:[4,5], 4:[6,7,8,9], 5:[], 6:[], 7:[], 8:[], 9:[]}
                }
        return self.graph_match(layer, graph)

    def opt_LayerConstSlice(self, layer):
        graph = self.get_matched_graph()
        shape = graph[6]
        ix = self.get_layers(shape.inputs[0])
        x = self.get_tensor(shape.name)
        layer.value = self.eval(layer, {x: np.asarray(ix.shape)})
        layer.op = 'Constant'
        del layer['inputs']

    def opt_IsLayerConstFill(self, layer):
        graph = { 'Sequence': {0:'Fill', 1:'Pack', 2:'Constant', 
                               3:'StridedSlice', 4:'Constant', 5:'StridedSlice', 6:'Constant',
                               7: 'Shape', 8:'Constant', 9:'Constant', 10:'Constant',
                               11:'Constant', 12:'Constant', 13:'Constant'},
                  'Connection': {0:[1,2], 1:[3,4,5,6], 2:[], 3:[7,8,9,10], 4:[], 5:[7,11,12,13], 6:[],
                                 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[]}
                }
        return self.graph_match(layer, graph)

    def opt_LayerConstFill(self, layer):
        graph = self.get_matched_graph()
        shape = graph[7]
        ix = self.get_layers(shape.inputs[0])
        x = self.get_tensor(shape.name)
        layer.value = self.eval(layer, {x: np.asarray(ix.shape)})
        layer.op = 'Constant'
        del layer['inputs']

    def opt_IsLayerConstMul(self, layer):
        graph = { 'Sequence': {0:'Mul', 1:'Range', 2:'StridedSlice', 
                               3:'Constant', 4:'StridedSlice', 5:'Constant',
                               6:'Shape', 7:'Constant', 8:'Constant', 9:'Constant',
                               10:'Constant', 11:'Constant', 12:'Constant',
                               },
                  'Connection': {0:[1,2], 1:[3,4,5], 2:[6,7,8,9], 4:[6,10,11,12]}
                }
        return self.graph_match(layer, graph)

    def opt_LayerConstMul(self, layer):
        graph = self.get_matched_graph()
        shape = graph[6]
        ix = self.get_layers(shape.inputs[0])
        x = self.get_tensor(shape.name)
        layer.value = self.eval(layer, {x: np.asarray(ix.shape)})
        layer.op = 'Constant'
        del layer['inputs']

    def opt_IsLayerReshapeAfterReshape(self, layer):
        r = False
        if(layer.op == 'Reshape'):
            inputs = self.get_layers(layer.inputs)
            if( (len(self.get_consumers(inputs[0])) == 1) and
                (inputs[0].op == 'Reshape')):
                r = True
        return r

    def opt_LayerReshapeAfterReshape(self, layer):
        inputs = self.get_layers(layer.inputs)
        self.opt_RemoveLayer(inputs[0])
        return True

    def opt_IsLayerReshapeNotNecesary(self, layer):
        r = False
        if(layer['op'] == 'Reshape'):
            inp = self.get_layers(layer.inputs[0])
            if(inp.shape == layer.shape):
                r = True
        return r

    def eval_shapes(self):
        shapes = self.kwargs['shapes']
        feed = {}
        for k, shape in shapes.items():
            layer = self.get_layers(k)
            if('dtype' in layer):
                feed[k] = np.zeros(shape, dtype=layer.dtype)
            else:
                feed[k] = np.zeros(shape)
            layer.shape = shape
        outputs = self.run([feed], model=self.lwnn_model)
        for k, v in outputs.items():
            if(v is not None):
                layer = self.get_layers(k)
                layer.shape = v.shape

    def run(self, feeds, **kwargs):
        outputs = {}
        otensors = []
        model = kwargs['model']
        for layer in model:
            if(layer.op != 'Output'):
                otensors.append(self.get_tensor(layer.name))
        for feed in feeds:
            onefeed={}
            for n, v in feed.items():
                onefeed[self.get_tensor(n)] = v
            outs = self.sess.run(otensors, onefeed)
            for i, v in enumerate(outs):
                n = model[i].name
                if(type(v) == bytes):
                    outputs[n] = None
                    continue
                elif(len(v.shape) == 0):
                    if(type(v) == np.ndarray): # bytes for wav input
                        print('warning: %s shape is empty'%(n))
                        outputs[n] = None
                        continue
                    else:
                        v= np.asanyarray([v])
                if(n in outputs):
                    outputs[n] = np.concatenate((outputs[n], v))
                else:
                    outputs[n] = v
        return outputs

    def convert2onnx(self):
        inputs = ['%s:0'%(layer.name) for layer in self.lwnn_model if layer.op == 'Input']
        if('output_node' in self.kwargs):
            outputs = ['%s:0'%(o) for o in self.kwargs['output_node']]
        else:
            outputs = ['%s:0'%(layer.name) for layer in self.lwnn_model if len(self.get_consumers(layer)) == 0]
        custom_ops = {}
        extra_opset = []
        graph_def = tf_optimize(inputs, outputs, self.graph_def, True)
        with tf.Graph().as_default() as tf_graph:
            tf.import_graph_def(graph_def, name='')
        with tf.Session(graph=tf_graph):
            g = process_tf_graph(tf_graph,
                                 continue_on_error=False,
                                 target=None,
                                 opset=None,
                                 custom_op_handlers=custom_ops,
                                 extra_opset=extra_opset,
                                 shape_override=None,
                                 input_names=inputs,
                                 output_names=outputs,
                                 inputs_as_nchw=None)

        onnx_graph = tf2onnx.optimizer.optimize_graph(g)
        model = onnx_graph.make_model(self.name)
        inps = []
        for inp in model.graph.input:
            if(inp.name in inputs):
                shape = [int(dim.dim_value) for dim in inp.type.tensor_type.shape.dim]
                if(len(shape) == 0):
                    layer = self.get_layers(self.LN(inp.name))
                    x = onnx.helper.make_tensor_value_info(inp.name, inp.type.tensor_type.elem_type, layer.shape)
                    inps.append(x)
                else:
                    inps.append(inp)
        outs = []
        for out in model.graph.output:
            if(out.name in inputs):
                shape = [int(dim.dim_value) for dim in out.type.tensor_type.shape.dim]
                if(len(shape) == 0):
                    layer = self.get_layers(self.LN(out.name))[0]
                    x = onnx.helper.make_tensor_value_info(out.name, out.type.tensor_type.elem_type, layer.shape)
                    inps.append(x)
                else:
                    inps.append(out)
        del model.graph.input[:]
        model.graph.input.extend(inps)
        del model.graph.output[:]
        model.graph.output.extend(outs)
        return model

    def handle_input_output(self):
        if('input_node' in self.kwargs):
            for inp in self.get_layers(self.kwargs['input_node']):
                inp.op = 'Input'
                del inp['inputs']
        if('output_node' in self.kwargs):
            for inp in self.get_layers(self.kwargs['output_node']):
                layer = LWNNLayer(name=inp.name+'_O',
                              op='Output',
                              inputs=[inp.name],
                              outputs=[inp.name+'_O'],
                              shape=inp.shape)
                self.lwnn_model.append(layer)

    def convert(self):
        self.lwnn_model = []
        for node in self.graph_def.node:
            if(node.op in ['Assert']): continue
            try:
                layer = self.to_LayerCommon(node)
            except Exception as e:
                raise Exception('failed to convert node %s: %s'%(node, e))
            self.lwnn_model.append(layer)
        if(('eval_shapes' in self.kwargs) and (self.kwargs['eval_shapes'] == True)):
            self.eval_shapes()
        if(('use_tf2onnx' not in self.kwargs) or (self.kwargs['use_tf2onnx'] == False)):
            self.handle_input_output()
            self.optimize(['graph_match'])
            self.optimize()
            self.remove_unused()
            for l in self.lwnn_model:
                if('shape' in l):
                    self.convert_layer_to_nchw(l)
            self.lwnn_modelo = self.clone()
            return
        # here a combination of tf2lwnn and tf2onnx to generate the lwnn model
        self.onnx_model = self.convert2onnx()
        shapes = {}
        for node in self.onnx_model.graph.node:
            name = self.LN(node.name)
            layer = self.get_layers([name])
            if(len(layer) > 0):
                layer = layer[0]
                if('shape' in layer):
                    shape = layer.shape
                    shapes[node.name] = shape
        converter = OnnxConverter(self.onnx_model, shapes=shapes)
        self.lwnn_modelo = converter.model

    @property
    def inputs(self):
        L = {}
        for layer in self.lwnn_model:
            if(layer.op == "Input"):
                L[layer.name] = layer
        return L

def tf2lwnn(graph_def, name, feeds=None, **kwargs):
    converter = TfConverter(graph_def, name, **kwargs)
    if(feeds != None):
        feeds = LWNNFeeder(feeds, converter.inputs, format='NHWC')
    model = LWNNModel(converter, name, feeds = feeds,
                      notPermuteReshapeSoftmax=True)
    model.generate()

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert tf to lwnn')
    parser.add_argument('-i', '--input', help='input tf model', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    parser.add_argument('-s', '--shape', help='shapes of some layers', nargs='+', default=None, required=False)
    parser.add_argument('--input_node', help='force which to be input node', nargs='+', default=None, required=False)
    parser.add_argument('--output_node', help='force which to be output node', nargs='+', default=None, required=False)
    parser.add_argument('--tf2onnx', help='if want to use tf2onnx instead of tf2lwnn', default=False, action='store_true', required=False)
    parser.add_argument('--dynamic_shape', help='dynamic shape support', default=False, action='store_true', required=False)
    parser.add_argument('--eval_shapes', help='eval shapes with zero inputs, used combined with --shape', default=False, action='store_true', required=False)
    parser.add_argument('--feeds', help='a json file describe the feeds in dict format, e.g: {"input":["/path/to/input1", "/path/to/input2"]}', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-3]
    kwargs = {}
    if((args.shape is not None) and (len(args.shape)%2 == 0)):
        n = int(len(args.shape)/2)
        shapes = {}
        for i in range(n):
            k = args.shape[2*i]
            shape = eval(args.shape[2*i+1])
            shapes[k] = shape
        kwargs['shapes'] = shapes
    kwargs['use_tf2onnx'] = args.tf2onnx
    kwargs['dynamic_shape'] = args.dynamic_shape
    kwargs['eval_shapes'] = args.eval_shapes
    if(args.input_node != None):
        kwargs['input_node'] = args.input_node
    if(args.output_node != None):
        kwargs['output_node'] = args.output_node
    tf2lwnn(args.input, args.output, args.feeds, **kwargs)
