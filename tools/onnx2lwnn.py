try:
    import onnx
except:
    print('onnx is not installed!')
import os

__all__ = ['onnx2lwnn']

def onnx2lwnn(model, name):
    if('/' not in name):
        p = 'models/%s'%(name)
    else:
        p = name
    if(not p.endswith('.c')):
        p = p + '.c'
    d = os.path.dirname(p)
    os.makedirs(d, exist_ok=True)
    print('generate %s'%(p))

    if(type(model) == str):
        model = onnx.load(model)
    #print(model.SerializeToString())
    fp = open(p, 'w')
    fp.write('#include "nn.h"')
    fp.close()
