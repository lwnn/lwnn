from keras.models import load_model,Model
import keras2onnx
from lwnn.core import *
from onnx2lwnn import *
import os
import json

__all__ = ['keras2lwnn']

class KerasConverter(LWNNUtil):
    def __init__(self, keras_model, **kwargs):
        self.opMap = {
            'InputLayer': 'Input',
            }
        self.keras_model = keras_model
        if('shape_infers' in kwargs):
            self.shapes = self.eval_shapes(kwargs['shape_infers'])
        else:
            self.shapes = {}
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
                if(type(inp) == list): inp=inp[0]
                inputs.append(inp)
            except ValueError:
                pass
        return inputs

    def get_outputs(self, klayer):
        outputs = []
        for i in range(1024):
            try:
                out = klayer.get_output_at(i)
                if(type(out) == list): out=out[0]
                outputs.append(out)
            except ValueError:
                pass
        return outputs

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
        for i, out in enumerate(outputs):
            shapes[out.name] = outs[i].shape
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
        layer = LWNNLayer(name=klayer.name, op=op, shape=shape, inputs=inputs, outputs=outputs, klweights=klayer.get_weights(), klconfig=klayer.get_config())
        return layer

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
                print(layer)
                self.lwnn_model.append(layer)

    def convert(self):
        self.lwnn_model = []
        for klayer in self.keras_model.layers:
            op = klayer.__class__.__name__
            if(op == 'Model'):
                self.convert_sub_model(klayer)
            else:
                layer = self.to_LayerCommon(klayer)
                print(layer)
                self.lwnn_model.append(layer)

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
        model = LWNNModel(converter, name, feeds = feeds,
                          notRmIdentity=True, notPermuteReshapeSoftmax=True)
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
