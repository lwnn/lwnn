
import keras
from keras.models import Model
from keras.models import load_model
from keras.layers import *
import numpy as np

from keras2lwnn import *

import os

os.environ['LWNN_GTEST'] = '1'

def conv2d(name, shape=[32,32,5], filters=24, kernel_size=(3,3), strides=(1,1), padding="same"):
    input = Input(shape=shape, name=name+'_input')
    weights = [np.random.uniform(low=-0.1,high=0.2,size=tuple(list(kernel_size)+[shape[-1],filters])).astype(np.float32),
               np.random.uniform(low=-0.1,high=0.2,size=tuple([filters])).astype(np.float32)]
    output = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                    weights = weights, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def relu(name, shape=[9,5,7]):
    input = Input(shape=shape, name=name+'_input')
    output = ReLU(name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def maxpool(name, shape=[32,32,3], pool_size=(2, 2), strides=(2, 2)):
    input = Input(shape=shape, name=name+'_input')
    output = MaxPooling2D(pool_size=pool_size, strides=strides, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def avgpool(name, shape=[13,17,3], pool_size=(2, 2), strides=(2, 2)):
    input = Input(shape=shape, name=name+'_input')
    output = AveragePooling2D(pool_size=pool_size, strides=strides, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def upsample2d(name, shape=[5,5,1], size=(2, 2)):
    input = Input(shape=shape, name=name+'_input')
    output = UpSampling2D(size=size, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.randint(low=-128,high=127,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def maxpool1d(name, shape=[32,32], pool_size=2, strides=2):
    input = Input(shape=shape, name=name+'_input')
    output = MaxPooling1D(pool_size=pool_size, strides=strides, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def avgpool1d(name, shape=[27,93], pool_size=2, strides=2):
    input = Input(shape=shape, name=name+'_input')
    output = AveragePooling1D(pool_size=pool_size, strides=strides, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def dense(name, row=8, units=1024):
    input = Input(shape=[row], name=name+'_input')
    weights = [np.random.uniform(low=-0.1,high=0.2,size=tuple([row, units])).astype(np.float32),
               np.random.uniform(low=-0.1,high=0.2,size=tuple([units])).astype(np.float32)]
    output = Dense(units, weights=weights, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10, row])).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def softmax(name, units=32):
    input = Input(shape=[units], name=name+'_input')
    output = Softmax(-1, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.array([float(i*i*i)/(units*units) for i in range(units)],dtype=np.float32).reshape(1,units)}
    keras2lwnn(model, name, feeds)

def pad(name, shape=[32,32,3], padding=(3,3)):
    input = Input(shape=shape, name=name+'_input')
    p = ZeroPadding2D(padding=padding)(input)
    output = Conv2D(3, kernel_size=(3,3), strides=(1,1), padding='same', name=name+'_output')(p)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=1,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def conv2d_bn(name, shape=[16,16,3], filters=10, kernel_size=(3,3), strides=(1,1), padding="same"):
    input = Input(shape=shape, name=name+'_input')
    weights = [np.random.uniform(low=-0.1,high=0.2,size=tuple(list(kernel_size)+[shape[-1],filters])).astype(np.float32),
               np.random.uniform(low=-0.1,high=0.2,size=tuple([filters])).astype(np.float32)]
    conv = Conv2D(filters, kernel_size=kernel_size, strides=strides, weights=weights, padding=padding)(input)
    output = BatchNormalization(name=name+'_output')(conv)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def conv1d(name, shape=[128,9], filters=32, kernel_size=9, strides=2, padding="same"):
    input = Input(shape=shape, name=name+'_input')
    weights = [np.random.uniform(low=-0.1,high=0.1,size=tuple([kernel_size,shape[-1],filters])).astype(np.float32),
               np.random.uniform(low=-0.1,high=0.1,size=tuple([filters])).astype(np.float32)]
    output = Conv1D(filters, kernel_size=kernel_size, strides=strides, padding=padding, weights=weights, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def dwconv2d(name, shape=[32,32,5], kernel_size=(3,3), strides=(1,1), padding="same"):
    input = Input(shape=shape, name=name+'_input')
    weights = [np.random.uniform(low=-0.1,high=0.2,size=tuple(list(kernel_size)+[shape[-1],1])).astype(np.float32),
               np.random.uniform(low=-0.1,high=0.2,size=tuple([shape[-1]])).astype(np.float32)]
    output = DepthwiseConv2D(kernel_size=kernel_size, strides=strides, padding=padding, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def concat(name, shape=[16, 16, 3], kernel_size=(3,3), strides=(1,1), padding="same", axis=-1):
    input = Input(shape=shape, name=name+'_input')
    x1 = Conv2D(32, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    x2 = Conv2D(32, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    x3 = Conv2D(32, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    output = concatenate([x1, x2,x3], axis=axis, name=name+'_output')
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=1,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def add(name, shape=[16, 16, 3], kernel_size=(3,3), strides=(1,1), padding="same"):
    input = Input(shape=shape, name=name+'_input')
    weights = [np.random.uniform(low=-0.1,high=0.5,size=tuple(list(kernel_size)+[shape[-1],32])).astype(np.float32),
               np.random.uniform(low=-0.1,high=3,size=tuple([32])).astype(np.float32)]
    x1 = Conv2D(32, kernel_size=kernel_size, strides=strides, weights=weights, padding=padding)(input)
    weights = [np.random.uniform(low=-0.7,high=0.9,size=tuple(list(kernel_size)+[shape[-1],32])).astype(np.float32),
               np.random.uniform(low=-0.1,high=2,size=tuple([32])).astype(np.float32)]
    x2 = Conv2D(32, kernel_size=kernel_size, strides=strides, weights=weights, padding=padding)(input)
    output = Add(name=name+'_output')([x1,x2])
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=1,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def transpose():
    O = 'models/transpose/golden'
    os.makedirs(O, exist_ok=True)
    for i,shape in enumerate([(1,2,4,3), (1,34,23,77)]):
        data = np.random.randint(low=0, high=100, size=shape).astype(np.float32)
        output = data.transpose(0,3,1,2)
        data.tofile('%s/input%s.raw'%(O, i))
        output.tofile('%s/output%s_0.raw'%(O, i))

def deconv2d(name, shape=[32,32,5], filters=24, kernel_size=(3,3), strides=(1,1), padding="same"):
    input = Input(shape=shape, name=name+'_input')
    weights = [np.random.uniform(low=-0.1,high=0.2,size=tuple(list(kernel_size)+[filters,shape[-1]])).astype(np.float32),
               np.random.uniform(low=-0.1,high=0.2,size=tuple([filters])).astype(np.float32)]
    output = Conv2DTranspose(filters, kernel_size=kernel_size, strides=strides, padding=padding,
                    weights = weights, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def bn(name, shape=[16,16,32]):
    C = shape[-1]
    input = Input(shape=shape, name=name+'_input')
    weights = [np.random.uniform(low=0.1,high=0.2,size=(C)).astype(np.float32),
               np.random.uniform(low=0.1,high=0.2,size=(C)).astype(np.float32),
               np.random.uniform(low=0.1,high=0.2,size=(C)).astype(np.float32),
               np.random.uniform(low=0.1,high=0.2,size=(C)).astype(np.float32),]
    output = BatchNormalization(name=name+'_output', weights = weights)(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=2,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def mnist():
    from keras.datasets import mnist
    from keras.utils import to_categorical
    #https://adventuresinmachinelearning.com/keras-tutorial-cnn-11-lines/
    def KModel(x_train, y_train, x_test, y_test):
        x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)/255.0
        x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.0
        y_train = to_categorical(y_train)
        y_test = to_categorical(y_test)
        x = Input(shape=x_train.shape[1:])

        conv1 = Conv2D(32, kernel_size=(5, 5), strides=(1, 1), padding='same')(x)
        h_conv1 = ReLU()(conv1)
        h_pool1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h_conv1)

        conv2 = Conv2D(64, (5, 5), strides=(1, 1), padding='same')(h_pool1)
        h_conv2 = ReLU()(conv2)
        h_pool2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(h_conv2)

        flat1 = Flatten()(h_pool2)
        fc1 = Dense(1024)(flat1)
        h_fc1 = ReLU()(fc1)

        fc2 = Dense(10)(h_fc1)

        y = Softmax()(fc2)

        model = Model(inputs=x, outputs=y)
        model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.SGD(lr=0.01), metrics=['accuracy'])
        model.fit(x_train, y_train, batch_size=128, epochs=12, verbose=1, validation_data=(x_test, y_test))
        return model
    if(os.path.exists('models/mnist/golden/output.raw')):
        print('mnist already generated, skip')
        return
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    if(os.path.exists('models/mnist.h5')):
        model = load_model('models/mnist.h5')
    else:
        model = KModel(x_train, y_train, x_test, y_test)
        model.save('models/mnist.h5')
    x_test = x_test.reshape(x_test.shape[0],x_test.shape[1],x_test.shape[2],1)/255.0
    x_test = x_test.astype(np.float32)
    y_test = y_test.astype(np.int32)
    keras2lwnn(model, 'mnist', {model.inputs[0]:x_test[0:100]})
    x_test.tofile('models/mnist/golden/input.raw')
    y_test.tofile('models/mnist/golden/output.raw')

def uci_inception():
    if(os.path.exists('models/uci_inception/golden/output.raw')):
        print('uci_inception already generated, skip')
        return
    if(os.path.exists('models/uci_inception/best_model.h5')):
        model = load_model('models/uci_inception/best_model.h5')
        x_test = np.fromfile('models/uci_inception/input.raw', dtype=np.float32).reshape(-1, 128, 9)
        y_test = np.fromfile('models/uci_inception/output.raw', dtype=np.int32)
        keras2lwnn(model, 'uci_inception', {model.inputs[0]:x_test[0:100]})
        x_test.tofile('models/uci_inception/golden/input.raw')
        y_test.tofile('models/uci_inception/golden/output.raw')

# https://keras-cn.readthedocs.io/en/latest/other/application/
def resnet50():
    from keras.applications import ResNet50
    model = ResNet50(weights='imagenet')
    keras2lwnn(model, 'resnet50')
def mobilenetv2():
    from keras.applications import MobileNetV2
    model = MobileNetV2(weights='imagenet')
    keras2lwnn(model, 'mobilenetv2')

if(__name__ == '__main__'):
    transpose()
    conv2d('conv2d_1',shape=[5,5,3], filters=1, kernel_size=(2,2), strides=(1,1), padding="same")
    conv2d('conv2d_2')
    conv2d('conv2d_3',shape=[45,17,23], filters=13, kernel_size=(2,3), strides=(3,2), padding="valid")
    relu('relu_1')
    maxpool('maxpool_1')
    maxpool('maxpool_2', shape=[30,20,5], pool_size=(3, 2), strides=(3, 2))
    dense('dense_1')
    dense('dense_2', 13, 1578)
    softmax('softmax_1')
    mnist()
    pad('pad_1')
    pad('pad_2',shape=[52,12,7], padding=(2,5))
    conv2d_bn('conv2dbn_1')
    conv1d('conv1d_1')
    dwconv2d('dwconv2d_1')
    dwconv2d('dwconv2d_2',shape=[57,15,3],kernel_size=(2,2), strides=(1,1), padding="same")
    dwconv2d('dwconv2d_3',shape=[45,17,23], kernel_size=(2,3), strides=(3,2), padding="valid")
    maxpool1d('maxpool1d_1')
    maxpool1d('maxpool1d_2',shape=[34,29], pool_size=3, strides=3)
    concat('concat_1')
    concat('concat_2', axis=1)
    concat('concat_3', axis=2)
    concat('concat_4', axis=0)
    uci_inception()
    avgpool('avgpool_1')
    avgpool('avgpool_2', shape=[37,240,5], pool_size=(2, 3), strides=(3, 1))
    avgpool1d('avgpool1d_1')
    avgpool1d('avgpool1d_2',shape=[341,129], pool_size=4, strides=5)
    add('add_1')
    upsample2d('upsample2d_1')
    upsample2d('upsample2d_2', shape=[8,8,3], size=(2, 3))
    upsample2d('upsample2d_3', shape=[19,19,13], size=(3, 2))
    deconv2d('deconv2d_1',shape=[5,5,3], filters=1, kernel_size=(2,2), strides=(2,2), padding="same")
    deconv2d('deconv2d_2')
    deconv2d('deconv2d_3',shape=[45,17,23], filters=13, kernel_size=(2,3), strides=(3,2), padding="valid")
    bn('bn_1')