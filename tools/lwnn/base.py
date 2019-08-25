# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import onnx
import onnxruntime
import os
import numpy as np

class LWNNBaseC():
    def __init__(self, model, T, feeds = None):
        self.GENL = {
                'Input': self.gen_LayerInput,
                'Conv': self.gen_LayerConv,
                'Relu': self.gen_LayerRelu,
                'MaxPool': self.gen_LayerMaxPool,
                'AveragePool': self.gen_LayerAveragePool,
                'Reshape': self.gen_LayerReshape,
                'Dense': self.gen_LayerDense,
                'Concat': self.gen_LayerConcat,
                'Pad': self.gen_LayerPad,
                'Softmax': self.gen_LayerSoftmax,
                'Add': self.gen_LayerAdd,
                'Output': self.gen_LayerOutput }
        self.model = model
        self.T = T
        self.feeds = feeds
        self.name = os.path.basename(self.model.name)

    def generate(self):
        self.fpH = self.model.open('%s.h'%(self.T))
        self.fpC = self.model.open('%s.c'%(self.T))
        self.fpC.write('#include "%s"\n\n'%(os.path.basename(self.fpH.name)))
        self.gen()
        self.model.close(self.fpH)
        self.model.close(self.fpC)
        if(('1' == os.getenv('LWNN_GTEST')) and(self.T == 'float')):
            self.gen_goldens_for_gtest()

    def gen_goldens_for_gtest(self):
        p = self.model.path
        p = os.path.abspath('%s/../golden'%(p))
        os.makedirs(p, exist_ok=True)
        if(self.feeds != None):
            feeds = {}
            for k,v in self.feeds.items():
                feeds[k] = v[0].reshape([1]+list(v[0].shape))
        else:
            feeds = None
        outputs = self.model.run(feeds)
        goldens = [n.name for n in self.model.onnx_model.graph.input] + \
                [n.name for n in self.model.onnx_model.graph.output]
        for n, v in outputs.items():
            if(self.model.is_model_channel_first()):
                if(len(v.shape) == 4):
                    v = v.transpose(0, 2, 3, 1)
                elif(len(v.shape) == 3):
                    v = v.transpose(0, 2, 1)
            if(n in goldens):
                # all gtest must be single input and single output
                if('input' in n):
                    n = 'input'
                else:
                    n = 'output'
                v.tofile('%s/%s.raw'%(p, n))

    def quantize(self, blob, only_needQ=False):
        min_value = np.min(blob)
        max_value = np.max(blob)
        if((min_value==0.0) and (max_value==0.0)):
            int_bits = 0
        else:
            int_bits = int(np.ceil(np.log2(max(abs(min_value), abs(max_value)))))
        if(self.T in ['q8', 's8']):
            dec_bits = 7 - int_bits
            dtype = np.int8
            cmax = 0x7F
            cmin = -0x80
        elif(self.T == 'q16'):
            # Note: here it was not 15, set to 8 to reduce the possiblity of
            # int32 overflow
            dec_bits = 8 - int_bits
            dtype = np.int16
            cmax = 0x7FFF
            cmin = -0x8000
        else:
            raise Exception('quantization is not supported for %s model\n'%(self.T))

        if(only_needQ==False):
            blobQ = np.clip(np.round(blob * 2 ** dec_bits), cmin, cmax).astype(dtype)
        else:
            blobQ = None

        return blobQ,dec_bits

    def to_nhwc(self, shape):
        if(len(shape)==4):
            # ONNX generally in format NCHW
            shape = [shape[i] for i in [0,2,3,1]]
        elif(len(shape)==3):
            shape = [shape[i] for i in [0,2,1]]
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
        if(self.T in ['q8', 's8', 'q16']):
            blobs = [self.get_Q_blob(layer)] + blobs
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

    def get_type(self):
        t = 'float'
        if(self.T == 'q8'):
            t = 'int8_t'
        elif(self.T == 'q16'):
            t = 'int16_t'
        return t

    def get_size(self, layer):
        sz = 1
        for s in layer['shape']:
            sz = sz*s
        return sz

    def gen_models(self):
        for layer in self.model.lwnn_model:
            if(layer['op'] == 'Input'):
                self.fpC.write('static %s %s_input_buffer[%s];\n'%(self.get_type(), layer['name'], self.get_size(layer)))
                self.fpC.write('static const nn_input_t %s_input=\n{\n\tL_REF(%s), %s_input_buffer\n};\n'
                               %(layer['name'],layer['name'],layer['name']))
        self.fpC.write('static const nn_input_t* const %s_%s_inputs[] =\n{\n'%(self.name, self.T))
        for layer in self.model.lwnn_model:
            if(layer['op'] == 'Input'):
                self.fpC.write('\t&%s_input,\n'%(layer['name']))
        self.fpC.write('\tNULL\n};\n\n')
        for layer in self.model.lwnn_model:
            if(layer['op'] == 'Output'):
                self.fpC.write('static %s %s_output_buffer[%s];\n'%(self.get_type(), layer['name'], self.get_size(layer)))
                self.fpC.write('static const nn_output_t %s_output=\n{\n\tL_REF(%s), %s_output_buffer\n};\n'
                               %(layer['name'],layer['name'],layer['name']))
        self.fpC.write('static const nn_output_t* const %s_%s_outputs[] =\n{\n'%(self.name, self.T))
        for layer in self.model.lwnn_model:
            if(layer['op'] == 'Output'):
                self.fpC.write('\t&%s_output,\n'%(layer['name']))
        self.fpC.write('\tNULL\n};\n\n')
        self.fpC.write('static const layer_t* const %s_%s_layers[] =\n{\n'%(self.name, self.T))
        for layer in self.model.lwnn_model:
            self.fpC.write('\tL_REF(%s),\n'%(layer['name']))
        self.fpC.write('\tNULL\n};\n\n')
        self.fpC.write('const network_t LWNN_%s_%s =\n{\n'%(self.name, self.T))
        self.fpC.write('\t"%s_%s",\n'%(self.name, self.T))
        self.fpC.write('\t%s_%s_layers,\n'%(self.name, self.T))
        self.fpC.write('\t%s_%s_inputs,\n'%(self.name, self.T))
        self.fpC.write('\t%s_%s_outputs,\n'%(self.name, self.T))
        self.fpC.write('\tNETWORK_TYPE_%s,\n'%(self.T.upper()))
        self.fpC.write('};\n\n')

    def gen_layer_common(self, layer):
        shape = self.get_shape(layer)
        self.fpC.write('#define %s_DIMS %s\n'%(layer['name'], 
                            ','.join(['%s'%(s) for s in shape])))

    def gen_LayerInput(self, layer):
        self.gen_no_blobs(layer)
        if(self.T in ['q8', 's8']):
            T = 'INT8'
        elif(self.T == 'q16'):
            T = 'INT16'
        else:
            T = 'FLOAT'
        self.fpC.write('L_INPUT ({0}, L_DT_{1});\n\n'.format(layer['name'], T))

    def gen_LayerConv(self, layer):
        raise NotImplementedError()
    def gen_LayerRelu(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_RELU ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerMaxPool(self, layer):
        if('pads' not in layer):
            pads = [0,0]
        else:
            pads = list(layer['pads'])
        M = np.asarray(list(layer['kernel_shape']) + pads + list(layer['strides']), np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('L_MAXPOOL ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerAveragePool(self, layer):
        if('pads' not in layer):
            pads = [0,0]
        else:
            pads = list(layer['pads'])
        M = np.asarray(list(layer['kernel_shape']) + pads + list(layer['strides']), np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('L_AVGPOOL ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerReshape(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_RESHAPE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def get_axis(self, layer):
        axis = layer['axis']
        shape = layer['shape']
        if(len(shape) == 4):
            axis = [0,3,1,2][axis]
        if(len(shape) == 3):
            axis = [0,3,1][axis]
        return axis

    def gen_LayerConcat(self, layer):
        M = np.asarray([self.get_axis(layer)], np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_CONCAT ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerDense(self, layer):
        raise NotImplementedError()

    def gen_LayerPad(self, layer):
        M = np.asarray(layer['pads'], np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('L_PAD ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerSoftmax(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_SOFTMAX ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerAdd(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_ADD ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerOutput(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_OUTPUT ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

