
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

if(__name__ == '__main__'):
    conv2d('conv2d_1',shape=[5,5,3], filters=1, kernel_size=(2,2), strides=(1,1), padding="same")
    conv2d('conv2d_2')
    conv2d('conv2d_3',shape=[45,17,23], filters=13, kernel_size=(2,3), strides=(3,2), padding="valid")
    relu('relu_1')
