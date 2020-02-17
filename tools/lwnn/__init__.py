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

