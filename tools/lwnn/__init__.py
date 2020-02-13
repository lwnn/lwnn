# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import glob
import numpy as np

def load_feeds(feeds_path, inputs):
    feeds_ = {}
    for rawF in glob.glob('%s/*.raw'%(feeds_path)):
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
    return feeds

def Layer2Str(layer):
    order = ['name', 'op', 'shape','inputs', 'outputs', 'weights','bias']
    def kv2s(k, v):
        cstr = ''
        try:
            cstr += '%s: %s, '%(k, v.shape)
        except:
            if(k in ['top', 'topq']):
                cstr += '%s: [ '%(k)
                for top in v:
                    try:
                        cstr += '%s, '%(str(top.shape))
                    except:
                        cstr += '%s, '%(top)
                cstr += '], '
            else:
                cstr += '%s: %s, '%(k,v)
        return cstr
    cstr = '{'
    for k in order:
        if(k in layer):
            cstr += kv2s(k, layer[k])
    for k,v in layer.items():
        if(k not in order):
            cstr += kv2s(k, v)
    cstr += '}\n'
    return cstr
