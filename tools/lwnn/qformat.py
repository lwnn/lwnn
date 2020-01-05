# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from .base import *

class LWNNQFormatC(LWNNBaseC):
    def __init__(self, model, T, feeds):
        try:
            super().__init__(model, T, feeds)
        except:
            LWNNBaseC.__init__(self, model, T, feeds)
        lwnn_model = self.model.clone()
        self.model.optimize(['RemoveReshape'])
        if(T == 's8'):
            self.model.optimize(['MergeReLUConv','MergeReLUDense'])
        self.calculate_output_encoding(feeds)
        self.fix_linked_to_the_same_Q()
        self.generate()
        self.model.set(lwnn_model)

    def calculate_output_encoding(self, feeds):
        self.output_encodings = {}
        self.outputs = self.model.run(feeds)
        for n,v in self.outputs.items():
            _,vq = self.quantize(v, True)
            self.output_encodings[n] = vq

    def get_encoding(self, layer, at=0):
        Q = self.output_encodings[layer['outputs'][at]]
        if('inputs' in layer):
            inputs = self.model.get_layers(layer['inputs'])
        if((layer['op'] == 'Softmax') or
           ((layer['op'] == 'Output') and 
            (len(inputs) == 1) and 
            (inputs[0]['op'] == 'Softmax'))):
            Q = eval(self.T[1:])-1
        return Q

    def is_QLayer(self, layer):
        r = False
        if(layer['op'] in ['Conv', 'Dense']):
            r = True
        return r

    def back_collect_tillQ(self, layer, linker, linked):
        if(linker['name'] not in linked):
            linked.append(linker['name'])
        for ly in self.model.get_layers(linker['inputs']):
            if(ly['name'] not in linked):
                linked.append(ly['name'])
                if(not self.is_QLayer(ly)):
                    self.back_collect_tillQ(ly, ly, linked)

    def fix_linked_to_the_same_Q(self):
        for layer in self.model.lwnn_model:
            Q = self.output_encodings[layer['outputs'][0]]
            linked = []
            consumers = self.model.get_consumers(layer)
            for c in consumers:
                if(c['op'] in ['Concat', 'Add']):
                    self.back_collect_tillQ(layer, c, linked)

            if(len(linked)>0):
                linked = self.model.get_layers(linked)
                self.set_linked_to_the_same_Q(linked, layer)
        for layer in self.model.lwnn_model:
            if(layer['op'] in ['Relu', 'MaxPool','AveragePool','Reshape','Output']):
                linked = self.model.get_layers(layer['inputs'])+[layer]
                self.set_linked_to_the_same_Q(linked[1:], linked[0], True)

    def set_linked_to_the_same_Q(self, linked, layer, sameAsLayer=False):
        Q = self.output_encodings[layer['outputs'][0]]
        for ly in linked:
            q = self.output_encodings[ly['outputs'][0]]
            if(q < Q):
                Q = q
        for ly in linked: # adjust all linked to the same Q
            q = self.output_encodings[ly['outputs'][0]]
            if((q != Q) and (sameAsLayer==False)):
                print('warning: linked %s, set Q from %s to %s, better do quantization awareness training to get the same Q'%(ly['name'], q, Q))
            self.output_encodings[ly['outputs'][0]] = Q

    def get_Q_blob(self, layer):
        return '%s_Q'%(layer['name']), np.asarray([self.get_encoding(layer)]).astype(np.int32)

    def gen_no_blobs(self, layer):
        self.gen_blobs(layer, [])

    def gen_LayerConv(self, layer):
        W = layer['weights']
        B = layer['bias']

        W,Wq = self.quantize(W)
        B,Bq = self.quantize(B)

        if('strides' not in layer):
            strides = [1, 1]
        else:
            strides = list(layer['strides'])

        M = np.asarray(list(layer['pads']) + strides + [Wq, Bq, self.get_activation(layer)], np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        if(layer['group'] == 1):
            op = 'CONV2D'
        elif(layer['group'] == layer['shape'][1]):
            op = 'DWCONV2D'
        else:
            raise Exception('convolution with group !=1 or !=C is not supported')
        self.fpC.write('L_{2} ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0], op))

    def gen_LayerConvTranspose(self, layer):
        W = layer['weights']
        B = layer['bias']

        W,Wq = self.quantize(W)
        B,Bq = self.quantize(B)

        if('strides' not in layer):
            strides = [1, 1]
        else:
            strides = list(layer['strides'])

        M = np.asarray(list(layer['pads']) + strides + [Wq, Bq, self.get_activation(layer)], np.int32)
        self.gen_layer_WBM(layer, W, B, M)

        self.fpC.write('L_DECONV2D ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def convert_to_x4_weights(self, weights):
        if(self.T in ['q8', 's8']):
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

        Wt = W.transpose(1,0)
        Wt,Wq = self.quantize(Wt)
        Wt = self.convert_to_x4_weights(Wt.reshape(Wt.shape[0],Wt.shape[1],1,1))
        B,Bq = self.quantize(B)

        M = np.asarray(list([Wq, Bq]), np.int8)

        self.gen_layer_WBM(layer, Wt.reshape(W.shape), B, M)

        self.fpC.write('L_DENSE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def get_LayerPadBlobs(self, layer):
        if('value' in layer):
            value = layer['value']
        else:
            value = 0
        if(self.T == 'q16'):
            dtype = np.int16
        else:
            dtype = np.int8
        Q = self.get_encoding(layer)
        return np.asarray([value*(2**Q)], dtype)

    def gen_LayerConst(self, layer):
        const = layer['const']
        if('ConcatOnPriorBox' in layer):
            pass
        else:
            const,constQ = self.quantize(const)
            Q = self.get_encoding(layer)
            assert(constQ == Q)
        self.gen_blobs(layer, [('%s_CONST'%(layer['name']), const)])
        self.fpC.write('L_CONST ({0});\n\n'.format(layer['name']))

    def gen_LayerBatchNormalization(self, layer):
        if('epsilon' in layer):
            epsilon = layer['epsilon']
        else:
            epsilon = 1e-05
        scale = layer['scale']
        var = layer['var']
        mean = layer['mean']
        bias = layer['bias']
        # y = s * (x - mean) / np.sqrt(var + epsilon) + bias
        # Y/2^Qy = s * (X/2^Qx - mean) / np.sqrt(var + epsilon) + bias
        # Y = s*2^Qy/2^Qx/np.sqrt(var + epsilon)*X - s*mean*2^Qy/np.sqrt(var + epsilon) + bias*2^Qy
        Qy = self.get_encoding(layer)
        inp = self.model.get_layers(layer['inputs'])[0]
        Qx = self.get_encoding(inp)
        scaleQ = scale*(2**Qy)/(2**Qx)/np.sqrt(var + epsilon)
        biasQ = -scale*mean*(2**Qy)/np.sqrt(var + epsilon) + bias*(2**Qy)
        scaleQ,Qs = self.quantize(scaleQ)
        biasQ,Qb = self.quantize(biasQ)
        # Y = scaleQ*X/2^Qs + biasQ/2^Qb
        M = np.asarray([Qs, Qb], dtype=np.int32)
        blobs=[('%s_scale'%(layer['name']),scaleQ),
               ('%s_bias'%(layer['name']),biasQ),
               ('%s_M'%(layer['name']),M)]
        self.gen_blobs(layer, blobs)
        self.fpC.write('L_BATCHNORM ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

class LWNNQSFormatC(LWNNQFormatC):
    def __init__(self, model, feeds):
        try:
            super().__init__(model, 's8', feeds)
        except:
            LWNNQFormatC.__init__(self, model, 's8', feeds)

    def quantize_QSZ(self, v):
        if(v is None): # layer fallback to float
            return None, 1, 7, 0
        min_value = np.min(v)
        max_value = np.max(v)
        if((min_value==0.0) and (max_value==0.0)):
            scale = 1
        else:
            middle = (min_value+max_value)/2
            min_value = min_value - middle
            max_value = max_value - middle
            scale = max_value/(127.0/(2**7))
        vq = 7 # always 7 fraction bits for lwnn
        cmax = 0x7F
        cmin = -0x80
        VQ = np.round(v/scale*(2**vq)).astype(np.int32)
        minq = np.min(VQ)
        maxq = np.max(VQ)
        if((minq >= cmin) and (maxq <=cmax)):
            Z = 0
        else:
            Z = np.round((maxq+minq)/2)
        VQ = np.clip(VQ-Z, cmin, cmax).astype(np.int8)
        return VQ, scale, vq, Z

    def calculate_output_encoding(self, feeds):
        self.output_encodings = {}
        self.output_offsets = {}
        self.output_scales = {}
        self.outputs = self.model.run(feeds)
        for n,v in self.outputs.items():
            _,scale,Q,Z = self.quantize_QSZ(v)
            self.output_offsets[n] = Z
            self.output_encodings[n] = Q
            self.output_scales[n] = scale

    def set_linked_to_the_same_Q(self, linked, layer, sameAsLayer=False):
        Q = self.output_encodings[layer['outputs'][0]]
        Z = self.get_offset(layer)
        S = self.get_scale(layer)
        sameQSZ = True
        for ly in linked:
            if(Q != self.get_encoding(ly)):
                sameQSZ = False
                break
            if(Z != self.get_offset(ly)):
                sameQSZ = False
                break
            if(S != self.get_scale(ly)):
                sameQSZ = False
                break
        if(sameQSZ):
            return
        if(sameAsLayer==False):
            bigV =[]
            for ly in linked:
                bigV.extend(self.outputs[ly['outputs'][0]].reshape(-1).tolist())
            bigV = np.asarray(bigV)
            _,S,Q,Z = self.quantize_QSZ(bigV)
        for ly in linked: # adjust all linked to the same Q
            q = self.output_encodings[ly['outputs'][0]]
            s = self.output_scales[ly['outputs'][0]]
            z = self.output_offsets[ly['outputs'][0]]
            aZ = Z
            if(ly['op'] == 'Add'):
                aZ = Z*2
            if(((q != Q) or (s != S) or (z != aZ)) and (sameAsLayer==False)):
                print('warning: linked %s, set Q:S:Z from %s : %.3f : %s to %s : %.3f : %s,\n'
                      '\tbetter do quantization awareness training to get the same Q'%(ly['name'], 
                        q,s,z,
                        Q, S, aZ))
            self.output_encodings[ly['outputs'][0]] = Q
            self.output_offsets[ly['outputs'][0]] = aZ
            self.output_scales[ly['outputs'][0]] = S

    def get_Q_blob(self, layer):
        return '%s_Q'%(layer['name']), np.asarray([self.get_encoding(layer), self.get_offset(layer), self.get_scaleQ(layer)]).astype(np.int32)

    def get_offset(self, layer, at=0):
        offset = self.output_offsets[layer['outputs'][at]]
        return offset

    def get_scale(self, layer, at=0):
        scale = self.output_scales[layer['outputs'][at]]
        return scale

    def scaleQ(self,scale):
        scale = int(scale*(1<<16))
        return scale

    def get_scaleQ(self, layer, at=0):
        # according to arm_nn_sat_doubling_high_mult
        scale = self.output_scales[layer['outputs'][at]]
        return self.scaleQ(scale)

    def gen_LayerConv(self, layer):
        # https://www.tensorflow.org/lite/performance/quantization_spec
        # real_value = (int8_value - zero_point) x scale
        # Conv2D:
        #   input : int8, per-tensor
        #   weights : int8, per-axis(0), zero_point=0
        #   bias: int32, per-axis(0), (scale, zero_point)=(input_scale*weight_scale[...], 0)
        #   output: int8, per-tensor
        # DwConv2D:
        #   input : int8, per-tensor
        #   weights : int8, per-axis(3), zero_point=0
        #   bias: int32, per-axis(3), (scale, zero_point)=(input_scale*weight_scale[...], 0)
        #   output: int8, per-tensor

        if(layer['group'] == 1):
            op = 'CONV2D'
        elif(layer['group'] == layer['shape'][1]):
            op = 'DWCONV2D'
        else:
            raise Exception('convolution with group !=1 or !=C is not supported')
        W = layer['weights']
        B = layer['bias']

        inp = self.model.get_layers(layer['inputs'])[0]

        Iq = self.get_encoding(inp)
        Oq = self.get_encoding(layer)
        Is = self.get_scale(inp)
        Os = self.get_scale(layer)

        filters = layer['shape'][1]
        OMult = np.ones(filters, dtype=np.int32)
        OShift = np.zeros(filters, dtype=np.int32)
        for i in range(filters):
            # TODO: scale for weights
            if(op == 'CONV2D'):
                W[i], Wq = self.quantize(W[i])
            else:
                W[:,:,:,i], Wq = self.quantize(W[:,:,:,i])
            OShift[i] = Wq+Iq-Oq
            OMult[i] = self.scaleQ(Is/Os)
            B[i] = B[i]*(2**(Iq+Wq))/Is

        W = W.astype(np.int8)
        B = B.astype(np.int32)

        if('strides' not in layer):
            strides = [1, 1]
        else:
            strides = list(layer['strides'])

        omin = -128
        if('activation' in layer):
            if(layer['activation'] == 'Relu'):
                omin = - self.get_offset(layer)

        M = np.asarray(list(layer['pads']) + strides + [omin], np.int32)
        n = layer['name']
        blobs = [('%s_W'%(n), W), ('%s_B'%(n), B), ('%s_M'%(n), M)]
        blobs.append(('%s_output_mult'%(n), OMult))
        blobs.append(('%s_output_shift'%(n), -OShift))
        self.gen_blobs(layer, blobs)

        self.fpC.write('L_{2} ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0], op))

    def gen_LayerConvTranspose(self, layer):
        W = layer['weights']
        B = layer['bias']

        inp = self.model.get_layers(layer['inputs'])[0]

        Iq = self.get_encoding(inp)
        Oq = self.get_encoding(layer)
        Is = self.get_scale(inp)
        Os = self.get_scale(layer)

        filters = layer['shape'][1]
        OMult = np.ones(filters, dtype=np.int32)
        OShift = np.zeros(filters, dtype=np.int32)
        for i in range(filters):
            W[i], Wq = self.quantize(W[i])
            OShift[i] = Wq+Iq-Oq
            OMult[i] = self.scaleQ(Is/Os)
            B[i] = B[i]*(2**(Iq+Wq))/Is

        W = W.astype(np.int8)
        B = B.astype(np.int32)

        if('strides' not in layer):
            strides = [1, 1]
        else:
            strides = list(layer['strides'])

        omin = -128
        if('activation' in layer):
            if(layer['activation'] == 'Relu'):
                omin = - self.get_offset(layer)

        M = np.asarray(list(layer['pads']) + strides + [omin], np.int32)
        n = layer['name']
        blobs = [('%s_W'%(n), W), ('%s_B'%(n), B), ('%s_M'%(n), M)]
        blobs.append(('%s_output_mult'%(n), OMult))
        blobs.append(('%s_output_shift'%(n), -OShift))
        self.gen_blobs(layer, blobs)

        self.fpC.write('L_DECONV2D ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def gen_LayerDense(self, layer):
        W = layer['weights']
        B = layer['bias']

        Wt = W.transpose(1,0)
        Wt,Ws,Wq,Wz = self.quantize_QSZ(Wt)

        inp = self.model.get_layers(layer['inputs'])[0]
        Iq = self.get_encoding(inp)
        Oq = self.get_encoding(layer)
        Is = self.get_scale(inp)
        Os = self.get_scale(layer)

        out_mult = self.scaleQ(Is*Ws/Os)

        B = B*(2**(Iq+Wq))/(Is*Ws)
        B = B.astype(np.int32)

        omin = -128
        if('activation' in layer):
            if(layer['activation'] == 'Relu'):
                omin = - self.get_offset(layer)

        M = np.asarray(list([Wq, Wz, out_mult, omin, self.get_activation(layer)]), np.int32)

        self.gen_layer_WBM(layer, Wt, B, M)

        self.fpC.write('L_DENSE ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))

    def get_QSZ(self, layer):
        Q = self.get_encoding(layer)
        Z = self.get_offset(layer)
        S = self.get_scale(layer)
        return Q,S,Z

    def get_LayerPadBlobs(self, layer):
        if('value' in layer):
            value = layer['value']
        else:
            value = 0
        Q,S,Z = self.get_QSZ(layer)
        return np.asarray([value/S*(2**Q)-Z], np.int8)

    def gen_LayerConst(self, layer):
        const = layer['const']
        if('ConcatOnPriorBox' in layer):
            pass
        else:
            const,constS,constQ,constZ  = self.quantize_QSZ(const)
            Q,S,Z = self.get_QSZ(layer)
            assert(constS == S)
            assert(constQ == Q)
            assert(constZ == Z)
        self.gen_blobs(layer, [('%s_CONST'%(layer['name']), const)])
        self.fpC.write('L_CONST ({0});\n\n'.format(layer['name']))

    def gen_LayerBatchNormalization(self, layer):
        if('epsilon' in layer):
            epsilon = layer['epsilon']
        else:
            epsilon = 1e-05
        scale = layer['scale']
        var = layer['var']
        mean = layer['mean']
        bias = layer['bias']
        # y = s * (x - mean) / np.sqrt(var + epsilon) + bias
        # Sy*(Y+Zy)/2^Qy = s * (Sx*(X+Zx)/2^Qx - mean) / np.sqrt(var + epsilon) + bias
        # Y = Sx/Sy*s*2^Qy/2^Qx/np.sqrt(var + epsilon)*X
        #      + Sx/Sy*s*2^Qy/2^Qx/np.sqrt(var + epsilon)*Zx
        #      - s*mean*2^Qy/Sy/np.sqrt(var + epsilon) + bias/Sy*2^Qy - Zy
        Qy,Sy,Zy = self.get_QSZ(layer)
        inp = self.model.get_layers(layer['inputs'])[0]
        Qx,Sx,Zx = self.get_QSZ(inp)
        scaleQ = Sx/Sy*scale*(2**Qy)/(2**Qx)/np.sqrt(var + epsilon)
        biasQ = Sx/Sy*scale*(2**Qy)/(2**Qx)/np.sqrt(var + epsilon)*Zx - scale*mean*(2**Qy)/Sy/np.sqrt(var + epsilon) + bias/Sy*(2**Qy) - Zy
        scaleQ,Qs = self.quantize(scaleQ)
        biasQ,Qb = self.quantize(biasQ)
        # Y = scaleQ*X/2^Qs + biasQ/2^Qb
        M = np.asarray([Qs, Qb], dtype=np.int32)
        blobs=[('%s_scale'%(layer['name']),scaleQ),
               ('%s_bias'%(layer['name']),biasQ),
               ('%s_M'%(layer['name']),M)]
        self.gen_blobs(layer, blobs)
        self.fpC.write('L_BATCHNORM ({0}, {1});\n\n'.format(layer['name'], layer['inputs'][0]))
