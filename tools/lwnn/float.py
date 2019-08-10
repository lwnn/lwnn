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
        if(len(W.shape)==4):
            W = W.transpose(0,2,3,1)
        B = layer['bias']

        M = np.asarray(list(layer['pads']) + list(layer['strides']), np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        self.fpC.write('L_CONV2D ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerDense(self, layer):
        W = layer['weights']
        W = W.transpose(1,0)
        B = layer['bias']

        self.gen_layer_WBM(layer, W, B)

        self.fpC.write('L_DENSE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerOutput(self, layer):
        self.gen_no_blobs(layer)
        self.fpC.write('L_OUTPUT ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))
