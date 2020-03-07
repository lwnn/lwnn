# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import glob
import numpy as np
import json

def load_feeds(feeds_path, inputs):
    feeds_ = {}
    for rawF in glob.glob('%s/*.raw'%(feeds_path)):
        raw = np.fromfile(rawF, np.float32)
        for n, shape in inputs.items():
            if(len(shape) == 4):
                shape = [shape[s] for s in [0,2,3,1]]
            else:
                raise
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


class LWNNFeeder():
    def __init__(self, feeds, inputs, **kwargs):
        if(type(feeds) == str):
            if(feeds.endswith('.json')):
                self._feeds = json.load(open(feeds, 'r'))
            elif(os.path.isdir(feeds)):
                self._feeds = load_feeds(feeds, inputs)
        else:
            self._feeds = feeds
        self._inputs = inputs
        self.kwargs = kwargs
        for _,v in self._feeds.items():
            if(type(v) == list):
                self._nbr = len(v)
            else:
                self._nbr = v.shape[0]
            break

    def isNHWC(self):
        if(('format' in self.kwargs) and (self.kwargs['format']=='NHWC')):
            return True
        return False

    def is_string_input(self, layer):
        if(('dtype' in layer) and (layer.dtype=='string')):
            return True
        else:
            return False

    def load(self, inp, layer):
        if(self.is_string_input(layer)):
            v = open(inp, 'rb').read()
        else:
            if(len(layer.shape) == 4):
                shape = [layer.shape[s] for s in [0,2,3,1]]
            else:
                raise
            if(self.isNHWC()):
                v = np.fromfile(inp, np.float32).reshape(shape)
            else:
                v = np.fromfile(inp, np.float32).reshape(shape)
                if(len(v.shape) == 4):
                    v = np.transpose(v, (0,3,1,2))
        return v

    def __iter__(self):
        for i in range(self._nbr):
            feed = {}
            for n, layer in self._inputs.items():
                inp = self._feeds[n][i]
                if(type(inp) == str):
                    feed[n] = self.load(inp, layer)
                else:
                    feed[n] = inp
            yield feed
