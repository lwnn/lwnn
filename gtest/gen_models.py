
import keras
from keras.models import Model
from keras.models import load_model
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
    y_test = y_test.astype(np.int8)
    keras2lwnn(model, 'mnist', {model.inputs[0]:x_test[0:100]})
    x_test.tofile('models/mnist/golden/input.raw')
    y_test.tofile('models/mnist/golden/output.raw')

if(__name__ == '__main__'):
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

