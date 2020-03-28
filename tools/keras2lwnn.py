from keras.models import load_model,Model
import keras2onnx
from lwnn.core import *
from onnx2lwnn import *
import os
import json

__all__ = ['keras2lwnn']

class KerasConverter(LWNNUtil):
    def __init__(self, keras_model, **kwargs):
        self.OPTIMIER = [
            (self.opt_IsLayerSliceBeforeROIAfterDetection, self.opt_LayerSliceBeforeROIAfterDetection, None),
            ]
        self.opMap = {
            'InputLayer': 'Input',
            'Conv2D': 'Conv',
            'Conv2DTranspose': 'ConvTranspose',
            'ZeroPadding2D': 'Pad',
            'BatchNorm': 'BatchNormalization',
            'Concatenate': 'Concat',
            'MaxPooling2D': 'MaxPool',
            'UpSampling2D': 'Upsample',
            'ProposalLayer': 'Proposal',
            'DetectionLayer': 'Detection',
            }
        self.TRANSLATOR = {
            'Input': self.to_LayerInput,
            'Identity': self.to_LayerIdentity,
            'Conv': self.to_LayerConv,
            'BatchNormalization': self.to_LayerBatchNormalization,
            'Activation': self.to_LayerActivation,
            'Lambda': self.to_LayerLambda,
            'Pad': self.to_LayerPad,
            'ConvTranspose': self.to_LayerConvTranspose,
            'Dense': self.to_LayerDense,
            'Concat': self.to_LayerConcat,
            'MaxPool': self.to_LayerMaxPool,
            'Proposal': self.to_LayerProposal,
            'Detection': self.to_LayerDetection,
             }
        self.keras_model = keras_model
        if('shape_infers' in kwargs):
            self.shapes = self.eval_shapes(kwargs['shape_infers'])
        else:
            self.shapes = {}
        self.kwargs = kwargs
        self.convert()

    @property
    def model(self):
        return self.lwnn_model

    def save(self, path):
        self.keras_model.save('%s.h5'%(path))

    def get_inputs(self, klayer):
        inputs = []
        for i in range(1024):
            try:
                inp = klayer.get_input_at(i)
                if(type(inp) == list):
                    inputs.extend(inp)
                else:
                    inputs.append(inp)
            except ValueError:
                pass
        return inputs

    def get_outputs(self, klayer):
        outputs = []
        for i in range(1024):
            try:
                out = klayer.get_output_at(i)
                if(type(out) == list):
                    outputs.extend(out)
                else:
                    outputs.append(out)
            except ValueError:
                pass
        return outputs

    def get_layer_inputs(self, layer):
        layers = []
        for name in layer.inputs:
            for ly in self.lwnn_model:
                for iname in ly.outputs:
                    if((iname==name) or (':'.join(iname.split(':')[:-1])==name)):
                        layers.append(ly)
        return layers

    def eval_shapes(self, feeds):
        CACHED_SHAPES='%s_cached_shapes.json'%(self.keras_model.name)
        if(os.path.exists(CACHED_SHAPES)):
            return json.load(open(CACHED_SHAPES, 'r'))
        shapes = {}
        outputs = []
        for klayer in self.keras_model.layers:
            op = klayer.__class__.__name__
            if(op in ['Model', 'InputLayer']):
                continue
            outputs.extend(self.get_outputs(klayer))
        model = Model(inputs=self.keras_model.input, outputs=outputs)
        outs = model.predict(feeds)
        for i, inp in enumerate(self.keras_model.input):
            shapes[inp.name] = feeds[i].shape
            #feeds[i].tofile('tmp/%s.raw'%(inp.name))
        for i, out in enumerate(outputs):
            shapes[out.name] = outs[i].shape
            #os.makedirs(os.path.dirname('tmp/%s.raw'%(out.name)), exist_ok=True)
            #outs[i].tofile('tmp/%s.raw'%(out.name))
        with open(CACHED_SHAPES, 'w') as f:
            json.dump(shapes, f)
        return shapes

    def to_LayerCommon(self, klayer):
        op = klayer.__class__.__name__
        if(op in self.opMap):
            op = self.opMap[op]
        inputs = [inp.name for inp in self.get_inputs(klayer)]
        outputs = [out.name for out in self.get_outputs(klayer)]
        if(outputs[0] in self.shapes):
            shape = self.shapes[outputs[0]]
        else:
            shape = klayer.get_output_shape_at(0)
        shape = list(shape)
        if(shape[0] is None):
            shape[0] = 1
        layer = LWNNLayer(name=klayer.name, op=op, shape=shape, inputs=inputs, outputs=outputs, klweights=klayer.get_weights(), klconfig=klayer.get_config())
        return layer

    def to_LayerInput(self, layer):
        del layer['inputs']

    def to_LayerIdentity(self, layer):
        if(None in layer.shape):
            inp = self.get_layers(layer.inputs[0])
            layer.shape = inp.shape

    def to_LayerConv(self, layer):
        klconfig = layer.klconfig
        klweights = layer.klweights
        layer.weights = W = klweights[0].transpose(3,2,0,1)
        layer.kernel_shape = [W.shape[2], W.shape[3]]
        if(len(klweights) == 2):
            layer.bias = klweights[1]
        layer.strides = klconfig['strides']
        layer.dilations = klconfig['dilation_rate']
        layer.activation = klconfig['activation']
        layer.padding = klconfig['padding'].upper()
        self.infer_conv_or_pool_shape_and_padding(layer)
        layer.group = 1

    def to_LayerConvTranspose(self, layer):
        self.to_LayerConv(layer)

    def to_LayerDense(self, layer):
        klweights = layer.klweights
        layer.weights = klweights[0]
        layer.bias = klweights[1]

    def to_LayerConcat(self, layer):
        klconfig = layer.klconfig
        layer.axis = klconfig['axis']
        if(None in layer.shape):
            dims = 0
            for inp in self.get_layers(layer.inputs):
                dims += inp.shape[layer.axis]
            layer.shape[layer.axis] = dims

    def to_LayerMaxPool(self, layer):
        klconfig = layer.klconfig
        layer.kernel_shape = klconfig['pool_size']
        layer.padding = klconfig['padding'].upper()
        layer.strides = klconfig['strides']
        self.infer_conv_or_pool_shape_and_padding(layer)

    def to_LayerProposal(self, layer):
        config = self.kwargs['rpnconfig']
        layer.RPN_BBOX_STD_DEV = config.RPN_BBOX_STD_DEV
        layer.RPN_ANCHOR_SCALES = config.RPN_ANCHOR_SCALES
        layer.RPN_ANCHOR_RATIOS = config.RPN_ANCHOR_RATIOS
        layer.BACKBONE_STRIDES = config.BACKBONE_STRIDES
        layer.IMAGE_SHAPE = config.IMAGE_SHAPE
        layer.RPN_ANCHOR_STRIDE = config.RPN_ANCHOR_STRIDE
        layer.RPN_NMS_THRESHOLD = config.RPN_NMS_THRESHOLD
        layer.inputs = layer.inputs[:2]

    def to_LayerDetection(self, layer):
        layer.shape[-1] = 7

    def to_LayerBatchNormalization(self, layer):
        klconfig = layer.klconfig
        klweights = layer.klweights
        layer.momentum = klconfig['momentum']
        layer.epsilon = klconfig['epsilon']
        layer.scale = klweights[0]
        layer.bias = klweights[1]
        layer.mean = klweights[2]
        layer.var = klweights[3]

    def to_LayerActivation(self, layer):
        klconfig = layer.klconfig
        activation = klconfig['activation']
        if(activation == 'relu'):
            layer.op = 'Relu'
        elif(activation == 'softmax'):
            layer.op = 'Softmax'
        else:
            raise Exception('%s: activation not supported'%(layer))
        if(None in layer.shape):
            inp = self.get_layers(layer.inputs[0])
            layer.shape = inp.shape

    def to_LayerLambda(self, layer):
        op1 = layer.outputs[0].split('/')[-1].split(':')[0]
        op = op1.split('_')[0]
        if(op in ['Reshape', 'Squeeze']):
            layer.op = op
            if(None in layer.shape):
                inp = self.get_layers(layer.inputs[0])
                dims = 1
                for s in inp.shape:
                    dims = dims*s
                for i,s in enumerate(layer.shape):
                    if(s != None):
                        dims = int(dims/s)
                    else:
                        axis = i
                layer.shape[axis] = dims
            # special handling for MaskRCNN
            consumers = self.get_consumers(layer)
            if((len(consumers) > 0) and all('TimeDistributed' in l for l in consumers)):
                layer.shape = layer.shape[1:]
        elif(op1 == 'strided_slice'):
            layer.op = 'Slice'
            inp = self.get_layers(layer.inputs[0])
            layer.starts = 0
            for i, (si, so) in enumerate(zip(inp.shape, layer.shape)):
                if(si != so):
                    layer.axes = i
                    layer.ends = so
                    break
        else:
            raise Exception('%s: not supported Lambda op %s'%(layer, op1))

    def to_LayerPad(self, layer):
        klconfig = layer.klconfig
        (padding_top, padding_bottom),(padding_left, padding_right) = klconfig['padding']
        layer.pads = [0,padding_top,padding_left,0, 0,padding_bottom,padding_right, 0]

    def convert_sub_model(self, submodel):
        # For MaskRcnn, the submodel 'rpn_model' repreated over 5 inputs
        inputs = self.get_inputs(submodel)
        for i,inp in enumerate(inputs[1:]):
            for klayer in submodel.layers:
                layer = self.to_LayerCommon(klayer)
                if(layer.op == 'Input'):
                    layer.op = 'Identity'
                    layer.inputs = [inp.name]
                else:
                    layer.inputs =  ['%s/%s:%s'%(submodel.name, x, i) for x in layer.inputs]
                layer.outputs =  ['%s/%s:%s'%(submodel.name, x, i) for x in layer.outputs]
                layer.name = '%s/%s:%s'%(submodel.name, layer.name, i)
                layer.submodel=submodel.name
                layer.submodel_id=i
                self.lwnn_model.append(layer)

    def convert_time_distributed(self, klayer):
        layer = self.to_LayerCommon(klayer)
        klconfig = layer.klconfig
        tlayer = klconfig['layer']
        op = tlayer['class_name']
        if(op in self.opMap):
            op = self.opMap[op]
        layer.op = op
        layer.TimeDistributed=True
        layer.klconfig=tlayer['config']
        self.lwnn_model.append(layer)

    def opt_IsLayerSliceBeforeROIAfterDetection(self, layer):
        r = False
        if(layer.op == 'Slice'):
            inp = self.get_layers(layer.inputs[0])
            consumers = self.get_consumers(layer)
            if((len(consumers)==1) and (inp.op=='Detection') and (consumers[0].op=='PyramidROIAlign')):
                r = True
        return r

    def opt_LayerSliceBeforeROIAfterDetection(self, layer):
        return self.opt_RemoveLayer(layer)

    def convert(self):
        self.lwnn_model = []
        for klayer in self.keras_model.layers:
            op = klayer.__class__.__name__
            if(op == 'Model'):
                self.convert_sub_model(klayer)
            elif(op == 'TimeDistributed'):
                self.convert_time_distributed(klayer)
            else:
                layer = self.to_LayerCommon(klayer)
                self.lwnn_model.append(layer)
        for out in self.get_outputs(self.keras_model):
            layer = LWNNLayer(name=out.name, op='Output', inputs=[out.name], outputs=[out.name+'_O'])
            self.lwnn_model.append(layer)
        for layer in self.lwnn_model:
            inputs = self.get_layer_inputs(layer)
            layer.inputs = [l.name for l in inputs]
            if(layer.op == 'Output'):
                layer.shape = inputs[0].shape
        for layer in self.lwnn_model:
            if(layer.op in self.TRANSLATOR):
                self.TRANSLATOR[layer.op](layer)
            if(('TimeDistributed' in layer) or (len(layer.shape) == 5)):
                layer.shape = layer.shape[1:]
            elif(('inputs' in layer) and all('TimeDistributed' in l for l in self.get_layers(layer.inputs))):
                layer.shape = layer.shape[1:]
            if(layer.op == 'Output'):
                inputs = self.get_layers(layer.inputs)
                layer.shape = inputs[0].shape
            else:
                del layer['klweights']
                del layer['klconfig']
        for layer in self.lwnn_model:
            layer.shape = [int(s) for s in layer.shape]
            self.convert_layer_to_nchw(layer)
        self.optimize()

def keras2lwnn(model, name, feeds=None, **kwargs):
    if(type(model) == str):
        model = load_model(model)
    os.makedirs('models/%s'%(name), exist_ok=True)
    try:
        from keras import backend as K
        import tensorflow as tf
        from tensorflow.python.framework import graph_util
        K.set_learning_phase(0)
        sess = K.get_session()
        outputs = [tf.identity(output, name=cstr(output.name)) for output in model.outputs]
        graph_def = graph_util.convert_variables_to_constants(sess,
                       sess.graph.as_graph_def(),
                       [output.name.split(':')[0] for output in outputs])
        with tf.gfile.FastGFile('models/%s/%s.pb'%(name,name), mode='wb') as f:
            f.write(graph_def.SerializeToString())
    except Exception as e:
        print(e)
    if(('use_keras2lwnn' in kwargs) and (kwargs['use_keras2lwnn'] == True)):
        converter = KerasConverter(model, **kwargs)
        if(feeds != None):
            feeds = LWNNFeeder(feeds, converter.inputs, format='NHWC')
        model = LWNNModel(converter, name, feeds = feeds, notPermuteReshapeSoftmax=True)
        model.generate()
        return
    onnx_model = keras2onnx.convert_keras(model, model.name,
                        channel_first_inputs=[model.input])
    onnx_feeds = {}
    if(feeds != None):
        for inp, v in feeds.items():
            onnx_feeds[inp.name] = v
    onnx2lwnn(onnx_model, name, onnx_feeds)
    if('1' == os.getenv('LWNN_GTEST')):
        model.save('models/%s/%s.h5'%(name,name))

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert keras to lwnn')
    parser.add_argument('-i', '--input', help='input keras model', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    parser.add_argument('--use_keras2lwnn', help='use keras2lwnn converter instead of keras2onnx&onnx2lwnn', default=False, action='store_true', required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-3]
    keras2lwnn(args.input, args.output)
