# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import pickle
import numpy as np
import glob
import torch
# ref https://pytorch.org/docs/stable/_modules/torch/nn/quantized/functional.html

__all__ = ['lwnn2torch']

class Lwnn2Torch():
    def __init__(self, p):
        self.lwnn_model = self.load(p)
        self.torch_model = self.convert()

    def load(self, p):
        try:
            model=pickle.load(open(p,'rb'))
        except:
            # load python2 generated pickle from python3
            model=pickle.load(open(p,'rb'), fix_imports=True, encoding="latin1")
        return model

    def convert(self):
        for ly in self.lwnn_model:
            print(ly['name'], ly['op'], ly['shape'])

def lwnn2torch(model, **kargs):
    if('feeds' in kargs):
        feeds = kargs['feeds']
    else:
        feeds = None

    if(type(feeds) == str):
        inputs = model.converter.inputs
        feeds_ = {}
        for rawF in glob.glob('%s/*.raw'%(feeds)):
            raw = np.fromfile(rawF, np.float32)
            for n, shape in inputs.items():
                if(len(shape) == 4):
                    shape = [shape[s] for s in [0,2,3,1]]
                sz = 1
                for s in shape:
                    sz = sz*s
                if(raw.shape[0] == sz):
                    raw = raw.reshape(shape)
                    if(n in feeds_):
                        feeds_[n] = np.concatenate((feeds_[n], raw))
                    else:
                        feeds_[n] = raw
                    print('using %s for input %s'%(rawF, n))
        feeds = {}
        for n,v in feeds_.items():
            if(len(v.shape) == 4):
                v = np.transpose(v, (0,3,1,2))
            feeds[n] = v
    tmodel = Lwnn2Torch(model)

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='convert lwnn to pytorch model')
    parser.add_argument('-i', '--input', help='input lwnn model', type=str, required=True)
    parser.add_argument('-r', '--raw', help='input raw directory', type=str, default=None, required=False)
    args = parser.parse_args()
    lwnn2torch(args.input, feeds=args.raw)
