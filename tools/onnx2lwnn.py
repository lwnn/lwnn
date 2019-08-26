# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

from lwnn import *
__all__ = ['onnx2lwnn']

def onnx2lwnn(model, name, feeds=None):
    '''
    feeds: mainly used to do quantization
    '''
    model = LWNNModel(model, name)
    model.gen_float_c(feeds)
    if(feeds != None):
        model.gen_quantized_c(feeds)


if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert onnx to lwnn')
    parser.add_argument('-i', '--input', help='input onnx model', type=str, required=True)
    parser.add_argument('-o', '--output', help='output lwnn model', type=str, default=None, required=False)
    args = parser.parse_args()
    if(args.output == None):
        args.output = os.path.basename(args.input)[:-5]
    onnx2lwnn(args.input, args.output)
