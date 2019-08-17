# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from .base import *

class LWNNQFormatC(LWNNBaseC):
    def __init__(self, model, T, feeds):
        super().__init__(model, T, feeds)
        lwnn_model = self.model.clone()
        self.model.optimize(['RemoveReshape'])
        self.output_encodings = self.calculate_output_encoding(feeds)
        self.generate()
        self.model.set(lwnn_model)

    def calculate_output_encoding(self, feeds):
        encodings = {}
        outputs = self.model.run(feeds)
        for n,v in outputs.items():
            _,vq = self.quantize(v, True)
            encodings[n] = vq
        return encodings

    def get_encoding(self, layer, at=0):
        Q = self.output_encodings[layer['outputs'][at]]
        if('inputs' in layer):
            inputs = self.model.get_layers(layer['inputs'])
        if((layer['op'] == 'Softmax') or
           ((layer['op'] == 'Identity') and 
            (len(inputs) == 1) and 
            (inputs[0]['op'] == 'Softmax'))):
            if(self.T == 'q8'):
                Q = 7
            elif(self.T == 'q16'):
                Q = 15
            else:
                assert(0)
        else:
            linked = []
            consumers = self.model.get_consumers(layer)
            for c in consumers:
                if(c['op'] == 'Concat'):
                    for ly in self.model.get_layers(c['inputs']):
                        if((ly['name'] not in linked) and (ly['name']!=layer['name'])):
                            linked.append(ly['name'])
            if(len(linked)>0):
                linked = self.model.get_layers(linked)
                for ly in linked:
                    q = self.output_encodings[ly['outputs'][0]]
                    if(q < Q):
                        Q = q
                for ly in linked: # adjust all linked to the same Q
                    self.output_encodings[ly['outputs'][0]] = Q
                self.output_encodings[layer['outputs'][at]] = Q
        return Q

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
        B = layer['bias']

        Oq = self.get_encoding(layer)
        W,Wq = self.quantize(W)
        B,Bq = self.quantize(B)

        if('strides' not in layer):
            strides = [1, 1]
        else:
            strides = list(layer['strides'])

        M = np.asarray(list(layer['pads']) + strides + [Wq, Bq, Oq], np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        if(layer['group'] == 1):
            op = 'CONV2D'
        elif(layer['group'] == layer['shape'][1]):
            op = 'DWCONV2D'
        else:
            raise Exception('convolution with group !=1 or !=C is not supported')
        self.fpC.write('L_{2} ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0], op))

    def convert_to_x4_weights(self, weights):
        if(self.T == 'q8'):
            return self.convert_to_x4_q7_weights(weights)
        else:
            return self.convert_to_x4_q15_weights(weights)
        
    def convert_to_x4_q7_weights(self, weights):
        [r, h, w, c] = weights.shape
        weights = np.reshape(weights, (r, h*w*c))
        num_of_rows = r
        num_of_cols = h*w*c
        new_weights = np.copy(weights)
        new_weights = np.reshape(new_weights, (r*h*w*c))
        counter = 0
        for i in range(int(num_of_rows/4)):
          # we only need to do the re-ordering for every 4 rows
          row_base = 4*i
          for j in range (int(num_of_cols/4)):
            # for each 4 entries
            column_base = 4*j
            new_weights[counter]   =  weights[row_base  ][column_base  ]
            new_weights[counter+1] =  weights[row_base+1][column_base  ]
            new_weights[counter+2] =  weights[row_base  ][column_base+2]
            new_weights[counter+3] =  weights[row_base+1][column_base+2]
            new_weights[counter+4] =  weights[row_base+2][column_base  ]
            new_weights[counter+5] =  weights[row_base+3][column_base  ]
            new_weights[counter+6] =  weights[row_base+2][column_base+2]
            new_weights[counter+7] =  weights[row_base+3][column_base+2]
    
            new_weights[counter+8] =  weights[row_base  ][column_base+1]
            new_weights[counter+9] =  weights[row_base+1][column_base+1]
            new_weights[counter+10] = weights[row_base  ][column_base+3]
            new_weights[counter+11] = weights[row_base+1][column_base+3]
            new_weights[counter+12] = weights[row_base+2][column_base+1]
            new_weights[counter+13] = weights[row_base+3][column_base+1]
            new_weights[counter+14] = weights[row_base+2][column_base+3]
            new_weights[counter+15] = weights[row_base+3][column_base+3]
            counter = counter + 16
          # the remaining ones are in order
          for j in range((int)(num_of_cols-num_of_cols%4), int(num_of_cols)):
            new_weights[counter] = weights[row_base][j]
            new_weights[counter+1] = weights[row_base+1][j]
            new_weights[counter+2] = weights[row_base+2][j]
            new_weights[counter+3] = weights[row_base+3][j]
            counter = counter + 4
        return new_weights

    def convert_to_x4_q15_weights(self, weights):
        [r, h, w, c] = weights.shape
        weights = np.reshape(weights, (r, h*w*c))
        num_of_rows = r
        num_of_cols = h*w*c
        new_weights = np.copy(weights)
        new_weights = np.reshape(new_weights, (r*h*w*c))
        counter = 0
        for i in range(int(num_of_rows/4)):
          # we only need to do the re-ordering for every 4 rows
          row_base = 4*i
          for j in range (int(num_of_cols/2)):
            # for each 2 entries
            column_base = 2*j
            new_weights[counter]   =  weights[row_base  ][column_base  ]
            new_weights[counter+1] =  weights[row_base  ][column_base+1]
            new_weights[counter+2] =  weights[row_base+1][column_base  ]
            new_weights[counter+3] =  weights[row_base+1][column_base+1]
            new_weights[counter+4] =  weights[row_base+2][column_base  ]
            new_weights[counter+5] =  weights[row_base+2][column_base+1]
            new_weights[counter+6] =  weights[row_base+3][column_base  ]
            new_weights[counter+7] =  weights[row_base+3][column_base+1]
    
            counter = counter + 8
          # the remaining ones are in order
          for j in range((int)(num_of_cols-num_of_cols%2), int(num_of_cols)):
            new_weights[counter] = weights[row_base][j]
            new_weights[counter+1] = weights[row_base+1][j]
            new_weights[counter+2] = weights[row_base+2][j]
            new_weights[counter+3] = weights[row_base+3][j]
            counter = counter + 4
        return new_weights

    def gen_LayerDense(self, layer):
        W = layer['weights']
        B = layer['bias']

        Oq = self.get_encoding(layer)
        Wt = W.transpose(1,0)
        Wt,Wq = self.quantize(Wt)
        Wt = self.convert_to_x4_weights(Wt.reshape(Wt.shape[0],Wt.shape[1],1,1))
        B,Bq = self.quantize(B)

        M = np.asarray(list([Wq, Bq, Oq]), np.int8)

        self.gen_layer_WBM(layer, Wt.reshape(W.shape), B, M)

        self.fpC.write('L_DENSE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerOutput(self, layer):
        blobs= [self.get_Q_blob(layer)]
        self.gen_blobs(layer, blobs)
        self.fpC.write('L_OUTPUT ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

