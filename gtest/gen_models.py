
from keras.models import Model
from keras.layers import *

from keras2lwnn import *

import os

os.environ['LWNN_GTEST'] = '1'

def conv2d(name, shape=[32,32,5], filters=24, kernel_size=(3,3), strides=(1,1), padding="same"):
    input = Input(shape=shape, name=name+'_input')
    output = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=1,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def relu(name, shape=[9,5,7]):
    input = Input(shape=shape, name=name+'_input')
    output = ReLU(name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=1,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def maxpool(name, shape=[32,32,3], pool_size=(2, 2), strides=(2, 2)):
    input = Input(shape=shape, name=name+'_input')
    output = MaxPooling2D(pool_size=pool_size, strides=strides, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=1,size=tuple([10]+shape)).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def dense(name, row=8, units=1024):
    input = Input(shape=[row], name=name+'_input')
    output = Dense(units, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.random.uniform(low=-1,high=1,size=tuple([10, row])).astype(np.float32)}
    keras2lwnn(model, name, feeds)

def softmax(name, units=32):
    input = Input(shape=[units], name=name+'_input')
    output = Softmax(units, name=name+'_output')(input)
    model = Model(inputs=input, outputs=output)
    feeds = {input:np.array([float(i*i*i)/(units*units) for i in range(units)],dtype=np.float32).reshape(1,units)}
    keras2lwnn(model, name, feeds)

if(__name__ == '__main__'):
#     conv2d('conv2d_1',shape=[5,5,3], filters=1, kernel_size=(2,2), strides=(1,1), padding="same")
#     conv2d('conv2d_2')
#     conv2d('conv2d_3',shape=[45,17,23], filters=13, kernel_size=(2,3), strides=(3,2), padding="valid")
#     relu('relu_1')
#     maxpool('maxpool_1')
#     maxpool('maxpool_2', shape=[30,20,5], pool_size=(3, 2), strides=(3, 2))
#     dense('dense_1')
#     dense('dense_2', 13, 1578)
    softmax('softmax_1')

