# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import os
import numpy as np

class LWNNBaseC():
    def __init__(self, model, T):
        self.GENL = {
                'Input': self.gen_LayerInput,
                'Conv': self.gen_LayerConv,
                'ConvTranspose': self.gen_LayerConvTranspose,
                'Relu': self.gen_LayerRelu,
                'Clip': self.gen_LayerClip,
                'PRelu': self.gen_LayerPRelu,
                'MaxPool': self.gen_LayerMaxPool,
                'Min': self.gen_LayerMin,
                'Mul': self.gen_LayerMul,
                'AveragePool': self.gen_LayerAveragePool,
                'Reshape': self.gen_LayerReshape,
                'Squeeze': self.gen_LayerReshape,
                'Dense': self.gen_LayerDense,
                'Concat': self.gen_LayerConcat,
                'Pad': self.gen_LayerPad,
                'Softmax': self.gen_LayerSoftmax,
                'Add': self.gen_LayerAdd,
                'Upsample': self.gen_LayerUpsample,
                'BatchNormalization': self.gen_LayerBatchNormalization,
                'Yolo': self.gen_LayerYolo,
                'YoloOutput': self.gen_LayerYoloOutput,
                'DetectionOutput': self.gen_LayerDetectionOutput,
                'Constant': self.gen_LayerConst,
                'Mfcc': self.gen_LayerMfcc,
                'LSTM': self.gen_LayerLSTM,
                'Transpose': self.gen_LayerTranspose,
                'Detection': self.gen_LayerDetection,
                'Proposal': self.gen_LayerProposal,
                'PyramidROIAlign': self.gen_LayerPyramidRoiAlign,
                'Slice': self.gen_LayerSlice,
                'Output': self.gen_LayerOutput }
        self.model = model
        self.T = T
        self.feeds = model.feeds
        self.name = os.path.basename(self.model.name)


    def open(self, fix='.c', flags='w'):
        if(fix != '.c'):
            fix = '_' + fix
        p = self.model.path + fix
        print('LWNN %s'%(p))
        fp = open(p, flags)
        return fp

    def close(self, fp):
        fp.close()

    def get_activation(self, layer):
        actMap = { 'linear':0, 'relu':1, 'leaky':2, 'sigmoid':3, 'tanh':4 }
        if('activation' not in layer):
            act = 0 # 0 means no activation
        else:
            act = actMap[layer['activation'].lower()]
        return act

    def generate(self):
        self.fpW = self.open('%s_builtin.h'%(self.T))
        self.fpB = self.open('%s.bin'%(self.T), 'wb')
        self.fpH = self.open('%s.h'%(self.T))
        self.fpC = self.open('%s.c'%(self.T))
        self.fpC.write('#include "nn.h"\n')
        self.fpC.write('#ifndef L_BLOB_NOT_BUILTIN\n')
        self.fpC.write('#include "%s"\n'%(os.path.basename(self.fpW.name)))
        self.fpC.write('#endif\n')
        self.fpC.write('#include "%s"\n\n'%(os.path.basename(self.fpH.name)))
        self.gen()
        self.close(self.fpH)
        self.close(self.fpC)
        self.close(self.fpW)
        self.close(self.fpB)
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
        goldens = [n.name for n in self.model.input] + \
                [n.name for n in self.model.output]
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
        if((blob is None) and (only_needQ==True)): # layer fallback to float
            return None,7
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
            blob = blob * (2**dec_bits)
            # Fix python2.7 numpy 'float' object has no attribute 'rint' issue
            blob = blob.astype(np.float32)
            blobQ = np.clip(np.round(blob), cmin, cmax).astype(dtype)
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

    def create_blobs_from_attrs(self, layer, attrs):
        n = layer.name
        blobs = []
        for k in attrs:
            V = layer[k]
            if(type(V) in [list, tuple]):
                if(all(isinstance(v, int) for v in V)):
                    blobs.append(('%s_%s'%(n,k), np.asarray(V, np.int32)))
                else:
                    blobs.append(('%s_%s'%(n,k), np.asarray(V, np.float32)))
            elif(type(V)==np.ndarray):
                if(V.dtype == np.float64):
                    V = V.astype(np.float32)
                elif(V.dtype == np.int64):
                    V = V.astype(np.int32)
                blobs.append(('%s_%s'%(n,k), V))
            elif(type(V)==int):
                blobs.append(('%s_%s'%(n,k), np.asarray([V], np.int32)))
            else:
                blobs.append(('%s_%s'%(n,k), np.asarray([V], np.float32)))
        return blobs

    def gen_blob(self, name, blob):
        T = self.get_blob_type(blob)
        self.fpW.write('#define l_blob_def_%s {'%(name))
        self.fpW.write(', '.join(['%s'%(f) for f in blob.reshape(-1)]))
        blob.tofile(self.fpB)
        self.fpW.write('}\n')
        self.fpH.write('#ifndef l_blob_def_%s\n'%(name))
        self.fpH.write('#define l_blob_def_%s (%s)\n'%(name, '*'.join(['%s'%(s) for s in blob.shape])))
        self.fpH.write('#endif\n')
        self.fpH.write('L_BLOB_DECLARE(%s, %s);\n'%(T, name))
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
        if((self.T in ['q8', 's8', 'q16']) and
           (layer['op'] not in ['DetectionOutput', 'Yolo', 'YoloOutput'])):
            # for float and those fallback to float layers, no Q blob
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
        elif((blob.dtype == np.int32) or (blob.dtype == np.int64)):
            return 'int32_t'
        elif(blob.dtype == np.uint32):
            return 'uint32_t'
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

    def get_type(self, layer):
        t = 'float'
        if(layer['op'] in ['DetectionOutput', 'YoloOutput']):
            pass # as only float version are supported
        elif(self.T in ['q8', 's8']):
            t = 'int8_t'
        elif(self.T == 'q16'):
            t = 'int16_t'
        if(layer.op == 'Mfcc'):
            t = 'void*' # for audio input
        return t

    def get_size(self, layer):
        sz = 1
        for s in layer['shape']:
            sz = sz*s
        if(layer.op == 'Mfcc'):
            sz = 2
        return sz

    def gen_models(self):
        for layer in self.model.lwnn_model:
            if(layer['op'] in ['Input', 'Mfcc']):
                self.fpC.write('static %s %s_input_buffer[%s];\n'%(self.get_type(layer), layer['name'], self.get_size(layer)))
                self.fpC.write('static const nn_input_t %s_input=\n{\n\tL_REF(%s), %s_input_buffer\n};\n'
                               %(layer['name'],layer['name'],layer['name']))
        self.fpC.write('static const nn_input_t* const %s_%s_inputs[] =\n{\n'%(self.name, self.T))
        for layer in self.model.lwnn_model:
            if(layer['op'] in ['Input', 'Mfcc']):
                self.fpC.write('\t&%s_input,\n'%(layer['name']))
        self.fpC.write('\tNULL\n};\n\n')
        for layer in self.model.lwnn_model:
            if((layer['op'] == 'Output') or ('Output' in layer)):
                if(self.get_size(layer) > 0):
                    self.fpC.write('static %s %s_output_buffer[%s];\n'%(self.get_type(layer), layer['name'], self.get_size(layer)))
                else:
                    self.fpC.write('#define %s_output_buffer NULL\n'%(layer.name))
                self.fpC.write('static const nn_output_t %s_output=\n{\n\tL_REF(%s), %s_output_buffer\n};\n'
                               %(layer['name'],layer['name'],layer['name']))
        self.fpC.write('static const nn_output_t* const %s_%s_outputs[] =\n{\n'%(self.name, self.T))
        for layer in self.model.lwnn_model:
            if((layer['op'] == 'Output') or ('Output' in layer)):
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
    def gen_LayerConvTranspose(self, layer):
        raise NotImplementedError()

    def gen_LayerRelu(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_RELU ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerClip(self, layer):
        if('min' in layer):
            mi = layer['min']
        else:
            mi = -np.inf
        if('max' in layer):
            mx = layer['max']
        else:
            mx = np.inf
        M = np.asarray([mi,mx], np.float32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('L_CLIP ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerPRelu(self, layer):
        weights = layer['weights']
        self.gen_blobs(layer, [('%s_W'%(layer['name']),weights)])
        self.fpC.write('L_PRELU ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerMaxPool(self, layer):
        if('pads' not in layer):
            pads = [0,0]
        else:
            pads = list(layer['pads'])[:2]
        with_mask = 0
        if(len(layer['outputs']) == 2):
            with_mask = 1
        M = np.asarray(list(layer['kernel_shape']) + pads + list(layer['strides']) + [with_mask], np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('L_MAXPOOL ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerMin(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'],
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_MINIMUM ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerMul(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'],
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_MUL ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerAveragePool(self, layer):
        if('pads' not in layer):
            pads = [0,0]
        else:
            pads = list(layer['pads'])[:2]
        with_mask = 0
        if(len(layer['outputs']) == 2):
            with_mask = 1
        M = np.asarray(list(layer['kernel_shape']) + pads + list(layer['strides']) + [with_mask], np.int32)
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
        elif(len(shape) == 3):
            axis = [0,3,1][axis]
        elif(len(shape) == 2):
            axis = [0,3][axis]
        else:
            assert(0)
        return axis

    def gen_LayerConcat(self, layer):
        M = np.asarray([self.get_axis(layer)], np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_CONCAT ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerDense(self, layer):
        raise NotImplementedError()

    def get_LayerPadBlobs(self, layer):
        if('value' in layer):
            value = layer['value']
        else:
            value = 0
        return np.asarray([value], np.float32)

    def gen_LayerPad(self, layer):
        M = np.asarray(layer['pads'], np.int32)
        P = self.get_LayerPadBlobs(layer)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M), 
                               ('%s_P'%(layer['name']),P)])
        self.fpC.write('L_PAD ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerSoftmax(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_SOFTMAX ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerAdd(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_ADD ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerUpsample(self, layer):
        self.gen_no_blobs(layer)
        if(len(layer['inputs']) == 2):
            self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
            self.fpC.write('L_UPSAMPLE2 ({0}, {0}_INPUTS);\n\n'.format(layer['name']))
        else:
            self.fpC.write('L_UPSAMPLE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerBatchNormalization(self, layer):
        if('momentum' in layer):
            momentum = layer['momentum']
        else:
            momentum = 0.9
        if('epsilon' in layer):
            epsilon = layer['epsilon']
        else:
            epsilon = 1e-05
        M = np.asarray([epsilon, momentum], dtype=np.float32)
        blobs=[('%s_scale'%(layer['name']),layer['scale']),
               ('%s_bias'%(layer['name']),layer['bias']),
               ('%s_var'%(layer['name']),layer['var']),
               ('%s_mean'%(layer['name']),layer['mean']),
               ('%s_M'%(layer['name']),M)]
        self.gen_blobs(layer, blobs)
        self.fpC.write('L_BATCHNORM ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerYolo(self, layer):
        mask = np.asarray(layer['mask'], np.int32)
        anchors = np.asarray(layer['anchors'], np.int32)
        M = np.asarray([layer['classes'], layer['num'], layer['jitter'], layer['ignore_thresh'], layer['truth_thresh']], np.float32)
        self.gen_blobs(layer, [('%s_mask'%(layer['name']),mask),
                               ('%s_anchors'%(layer['name']),anchors),
                               ('%s_M'%(layer['name']),M)])
        self.fpC.write('L_YOLO ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerYoloOutput(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_YOLOOUTPUT ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerDetectionOutput(self, layer):
        M1 = np.array([layer['nms_threshold'], layer['confidence_threshold']], np.float32)
        M2 = np.array([layer['num_classes'], layer['share_location'], 
                       layer['background_label_id'], layer['top_k'], 
                       layer['keep_top_k'], layer['code_type']], np.int32)
        priorbox = layer['priorbox']
        self.gen_blobs(layer, [('%s_M1'%(layer['name']),M1), 
                           ('%s_M2'%(layer['name']),M2),
                           ('%s_priorbox'%(layer['name']),priorbox),])
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_DETECTIONOUTPUT ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerConst(self, layer):
        raise NotImplementedError()

    def gen_LayerMfcc(self, layer):
        M = np.asarray([layer.magnitude_squared, layer.window_size,
                        layer.stride, layer.desired_samples, layer.desired_channels,
                        layer.upper_frequency_limit, layer.lower_frequency_limit, 
                        layer.dct_coefficient_count, layer.filterbank_channel_count], 
                        np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('L_MFCC ({0});\n\n'.format(layer['name']))

    def gen_LayerLSTM(self, layer):
        raise NotImplementedError()

    def gen_LayerDetection(self, layer):
        raise NotImplementedError()

    def gen_LayerProposal(self, layer):
        raise NotImplementedError()

    def gen_LayerPyramidRoiAlign(self, layer):
        raise NotImplementedError()

    def gen_LayerSlice(self, layer):
        M = np.asarray([layer.starts, layer.ends, layer.axes], np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('L_SLICE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerTranspose(self, layer):
        perm = list(layer.perm)
        if(len(perm) == 3):
            if(perm == [0,2,1]):
                ptype = 0
            else:
                raise
        elif(len(perm) == 4):
            if(perm == [0,3,1,2]): # from NHWC to NCHW
                ptype = 0
            elif(perm == [0,2,3,1]): # from NCHW to NHWC
                ptype = 0x8000
            else:
                raise
        else:
            raise
        M = np.asarray([ptype], np.int32)
        self.gen_blobs(layer, [('%s_M'%(layer['name']),M)])
        self.fpC.write('L_TRANSPOSE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerOutput(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_OUTPUT ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

