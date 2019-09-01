# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from .base import *

class LWNNFloatC(LWNNBaseC):
    def __init__(self, model, feeds=None):
        try:
            super().__init__(model, 'float', feeds)
        except:
            LWNNBaseC.__init__(self, model, 'float', feeds)
        self.generate()

    def gen_LayerConv(self, layer):
        W = layer['weights']
        B = layer['bias']

        if('strides' not in layer):
            strides = [1, 1]
        else:
            strides = list(layer['strides'])

        M = np.asarray(list(layer['pads']) + strides, np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        if(layer['group'] == 1):
            op = 'CONV2D'
        elif(layer['group'] == layer['shape'][1]):
            op = 'DWCONV2D'
        else:
            raise Exception('convolution with group !=1 or !=C is not supported')

        self.fpC.write('L_{2} ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0], op))

    def gen_LayerDense(self, layer):
        W = layer['weights']
        W = W.transpose(1,0)
        B = layer['bias']

        self.gen_layer_WBM(layer, W, B)

        self.fpC.write('L_DENSE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerPriorBox(self, layer):
        M1 = np.array([layer['min_size'], layer['aspect_ratio'], layer['offset']] + layer['variance'], np.float32)
        M2 = np.array([layer['flip'], layer['clip']], np.int8)
        self.gen_blobs(layer, [('%s_M1'%(layer['name']),M1), 
                           ('%s_M2'%(layer['name']),M2)])
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_PRIORBOX ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerDetectionOutput(self, layer):
        M1 = np.array([layer['nms_threshold'], layer['confidence_threshold']], np.float32)
        M2 = np.array([layer['num_classes'], layer['share_location'], 
                       layer['background_label_id'], layer['top_k'], 
                       layer['keep_top_k'], layer['code_type']], np.int32)
        self.gen_blobs(layer, [('%s_M1'%(layer['name']),M1), 
                           ('%s_M2'%(layer['name']),M2)])
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_DETECTIONOUTPUT ({0}, {0}_INPUTS);\n\n'.format(layer['name']))
