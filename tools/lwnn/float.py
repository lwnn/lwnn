# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from .base import *

class LWNNFloatC(LWNNBaseC):
    def __init__(self, model):
        super().__init__(model, 'float')
        self.generate()

    def gen_LayerInput(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_INPUT ({0}, L_DT_FLOAT);\n\n'.format(layer['name']))

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

    def gen_LayerAdd(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('#define {0}_INPUTS {1}\n'.format(layer['name'], 
                        ','.join(['L_REF(%s)'%inp for inp in layer['inputs']])))
        self.fpC.write('L_ADD ({0}, {0}_INPUTS);\n\n'.format(layer['name']))

    def gen_LayerOutput(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_OUTPUT ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))
