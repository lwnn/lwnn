
import onnx
import onnxruntime
import os
import numpy as np

__all__ = ['onnx2lwnn']

class LWNNBaseC():
    def __init__(self, model, T):
        self.GENL = {
                'Input': self.gen_LayerInput,
                'Conv': self.gen_LayerConv,
                'Identity': self.gen_LayerOutput }
        self.model = model
        self.T = T
        self.name = os.path.basename(self.model.name)

    def generate(self):
        self.fpH = self.model.open('%s.h'%(self.T))
        self.fpC = self.model.open('%s.c'%(self.T))
        self.fpC.write('#include "%s"\n\n'%(os.path.basename(self.fpH.name)))
        self.gen()
        self.model.close(self.fpH)
        self.model.close(self.fpC)
        if('1' == os.getenv('LWNN_GTEST')):
            self.gen_goldens_for_gtest()

    def gen_goldens_for_gtest(self):
        p = self.model.path
        p = os.path.abspath('%s/../golden'%(p))
        os.makedirs(p, exist_ok=True)
        outputs = self.model.run()
        goldens = [n.name for n in self.model.onnx_model.graph.input] + \
                [n.name for n in self.model.onnx_model.graph.output]
        for n, v in outputs.items():
            if(self.model.is_model_channel_first()):
                if(len(v.shape) == 4):
                    v = v.transpose(0, 2, 3, 1)
            if(n in goldens):
                v.tofile('%s/%s.raw'%(p, n))

    def quantize(self, blob, only_needQ=False):
        min_value = np.min(blob)
        max_value = np.max(blob)
        if((min_value==0.0) and (max_value==0.0)):
            int_bits = 0
        else:
            int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
        if(self.T == 'q8'):
            dec_bits = 7 - int_bits
            dtype = np.int8
        elif(self.T == 'q16'):
            dec_bits = 15 - int_bits
            dtype = np.int16
            blobQ = np.round(blob * 2 ** dec_bits).astype(np.int16)
        else:
            raise Exception('quantization is not supported for %s model\n'%(self.T))

        if(only_needQ==False):
            blobQ = np.round(blob * 2 ** dec_bits).astype(dtype)
        else:
            blobQ = None

        return blobQ,dec_bits

    def to_nhwc(self, shape):
        if(len(shape)==4):
            shape = [shape[i] for i in [0,2,3,1]]
        return shape

    def get_shape(self, layer):
        shape = layer['shape']
        return self.to_nhwc(shape)

    def gen(self):
        self.gen_layers()
        self.gen_models()

    def gen_blob(self, name, blob):
        T = self.get_blob_type(blob)
        self.fpH.write('static const %s %s[] = {'%(T, name))
        self.fpH.write(', '.join(['%s'%(f) for f in blob.reshape(-1)]))
        self.fpH.write('};\n')
        self.fpH.write('static const int l_dims_%s[]={ %s,0 };\n'%(
            name, ','.join(['%s'%(s) for s in blob.shape])))
        if(T.endswith('_t')):
            T=T[:-2]
        self.fpH.write('static const layer_blob_t l_blob_%s =\n{\n'%(name))
        self.fpH.write('\tl_dims_%s,\n'%(name))
        self.fpH.write('\tL_DT_%s,\n'%(T.upper()))
        self.fpH.write('\t(void*)%s,\n'%(name))
        self.fpH.write('};\n')

    def gen_blobs(self, layer, blobs):
        for blob in blobs:
            self.gen_blob(*blob)
        self.fpH.write('static const layer_blob_t* l_blobs_%s[] =\n{\n'%(layer['name'])) 
        for name, _ in blobs:
            self.fpH.write('\t&l_blob_%s,\n'%(name))
        self.fpH.write('\tNULL\n};\n\n')

    def gen_no_blobs(self, layer):
        self.fpC.write('#define l_blobs_%s NULL\n'%(layer['name']))

    def get_blob_type(self, blob):
        if(blob.dtype == np.float32):
            return 'float'
        elif(blob.dtype == np.int32):
            return 'int32_t'
        elif(blob.dtype == np.int16):
            return 'int16_t'
        elif(blob.dtype == np.int8):
            return 'int8_t'
        raise Exception('unsupported numpy type %s'%(blob.dtype))

    def gen_layer_WBM(self, layer, W, B, M=None):
        n = layer['name']
        blobs = [('%s_W'%(n), W), ('%s_B'%(n), B)]
        if(type(M) == np.ndarray):
            blobs.append(('%s_M'%(n), M))
        self.gen_blobs(layer, blobs)

    def gen_layers(self):
        for layer in self.model.lwnn_model:
            self.gen_layer_common(layer)
            self.GENL[layer['op']](layer)

    def gen_models(self):
        self.fpC.write('static const layer_t* const %s_%s_layers[] =\n{\n'%(self.name, self.T))
        for layer in self.model.lwnn_model:
            self.fpC.write('\tL_REF(%s),\n'%(layer['name']))
        self.fpC.write('\tNULL\n};\n\n')
        self.fpC.write('const network_t LWNN_%s_%s =\n{\n'%(self.name, self.T))
        self.fpC.write('\t"%s_%s",\n'%(self.name, self.T))
        self.fpC.write('\t%s_%s_layers\n'%(self.name, self.T))
        self.fpC.write('};\n\n')

    def gen_layer_common(self, layer):
        shape = self.get_shape(layer)
        self.fpC.write('#define %s_DIMS %s\n'%(layer['name'], 
                            ','.join(['%s'%(s) for s in shape])))

    def gen_LayerInput(self, layer):
        raise NotImplementedError()
    def gen_LayerConv(self, layer):
        raise NotImplementedError()
    def gen_LayerOutput(self, layer):
        raise NotImplementedError()

class LWNNFloatC(LWNNBaseC):
    def __init__(self, model):
        super().__init__(model, 'float')
        self.generate()

    def gen_LayerInput(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_INPUT ({0}, L_DT_FLOAT);\n\n'.format(layer['name']))

    def gen_LayerConv(self, layer):
        W = layer['weights']
        if(len(W.shape)==4):
            W = W.transpose(0,2,3,1)
        B = layer['bias']

        M = np.asarray(list(layer['pads']) + list(layer['strides']), np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        self.fpC.write('L_CONV2D ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerOutput(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_OUTPUT ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

class LWNNQFormatC(LWNNBaseC):
    def __init__(self, model, feeds, T):
        super().__init__(model, T)
        self.output_encodings = self.calculate_output_encoding(feeds)
        self.generate()

    def calculate_output_encoding(self, feeds):
        encodings = {}
        outputs = self.model.run(feeds)
        for n,v in outputs.items():
            _,vq = self.quantize(v, True)
            encodings[n] = vq
        return encodings

    def get_encoding(self, layer, at=0):
        return self.output_encodings[layer['outputs'][at]]

    def get_Q_blob(self, layer):
        return '%s_Q'%(layer['name']), np.asarray([self.get_encoding(layer)]).astype(np.int8)

    def gen_LayerInput(self, layer):
        blobs= [self.get_Q_blob(layer)]
        self.gen_blobs(layer, blobs)
        if(self.T == 'q8'):
            T = 'INT8'
        elif(self.T == 'q16'):
            T = 'INT16'
        self.fpC.write('L_INPUT ({0}, L_DT_{1});\n\n'.format(layer['name'],T))

    def gen_LayerConv(self, layer):
        W = layer['weights']
        if(len(W.shape)==4):
            W = W.transpose(0,2,3,1)
        B = layer['bias']

        Oq = self.get_encoding(layer)
        W,Wq = self.quantize(W)
        B,Bq = self.quantize(B)

        M = np.asarray(list(layer['pads']) + list(layer['strides']) + [Wq, Bq, Oq], np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        self.fpC.write('L_CONV2D ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerOutput(self, layer):
        blobs= [self.get_Q_blob(layer)]
        self.gen_blobs(layer, blobs)
        self.fpC.write('L_OUTPUT ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))


class LWNNModel():
    def __init__(self, onnx_model, name):
        self.TRANSLATOR = {
                    'Transpose': self.to_LayerTranspose,
                    'Conv': self.to_LayerConv,
                    'Identity': self.to_LayerIdentity }
        self.is_model_channel_first_cached=None
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
        LWNNQFormatC(self, feeds, 'q8')
        LWNNQFormatC(self, feeds, 'q16')

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
        if('pads' not in layer):
            layer['pads'] = [0,0,0,0]
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
                     'outputs' : [inp.name],
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

    def is_model_channel_first(self):
        if(self.is_model_channel_first_cached != None):
            return self.is_model_channel_first_cached
        r = True
        for layer in self.lwnn_model:
            if(self.is_input_channel_adjusted(layer)):
                r = False
        self.is_model_channel_first_cached = r
        return r

    def remove_adjust_layer(self):
        if(self.is_model_channel_first()):
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

def onnx2lwnn(model, name, feeds=None):
    '''
    feeds: mainly used to do quantization
    '''
    model = LWNNModel(model, name)
    model.gen_float_c()
    if(feeds != None):
        model.gen_quantized_c(feeds)

