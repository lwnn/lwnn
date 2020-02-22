from keras.models import load_model
import keras2onnx
from onnx2lwnn import *
import os

__all__ = ['keras2lwnn']

def keras2lwnn(model, name, feeds=None):
    if(type(model) == str):
        model = load_model(model)
    try:
        from keras import backend as K
        import tensorflow as tf
        sess = K.get_session()
        with tf.gfile.FastGFile('models/%s/%s.pb'%(name,name), mode='wb') as f:
            f.write(sess.graph_def.SerializeToString())
    except Exception as e:
        print(e)
    onnx_model = keras2onnx.convert_keras(model, model.name,
                        channel_first_inputs=[model.input])
    onnx_feeds = {}
    if(feeds != None):
        for inp, v in feeds.items():
            onnx_feeds[inp.name] = v
    onnx2lwnn(onnx_model, name, onnx_feeds)
    if('1' == os.getenv('LWNN_GTEST')):
        os.makedirs('models/%s'%(name), exist_ok=True)
        model.save('models/%s/%s.h5'%(name,name))

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert keras to lwnn')
    parser.add_argument('-i', '--input', help='input keras model', type=str, required=True)
    parser.add_argument('-o', '--output', help='output keras model', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-3]
    keras2lwnn(args.input, args.output)
