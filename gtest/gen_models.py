
from keras.models import Model
from keras.layers import *

from keras2lwnn import *

def conv2d(name, shape=[32,32,3], filters=24, kernel_size=(3,3), strides=(1,1), padding="same"):
    input = Input(shape=shape)
    output = Conv2D(filters, kernel_size=kernel_size, strides=strides, padding=padding)(input)
    model = Model(inputs=input, outputs=output)
    keras2lwnn(model, name)

if(__name__ == '__main__'):
    conv2d('conv2d_1')

