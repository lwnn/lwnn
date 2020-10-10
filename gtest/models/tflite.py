import sys
import os
import re
LWNN=os.path.abspath('../..')
if(not os.path.isfile('%s/Console.bat'%(LWNN))):
    LWNN=os.path.abspath('.')
p=os.path.abspath('%s/tools'%(LWNN))
sys.path.append(p)
import numpy as np
from verifyoutput import *
import glob
import liblwnn as lwnn
import tensorflow as tf
lwnn.set_log_level(0)

for tfmodel in glob.glob('*/*quantized.tflite'):
    print('eval for:', tfmodel)
    if(any(v in tfmodel for v in ['concat_4'])):
        print('  skip')
        continue
    try:
        DIR = os.path.dirname(tfmodel)
        model = lwnn.TfLite(tfmodel)
        interpreter = tf.lite.Interpreter(model_path=tfmodel)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        data = np.fromfile('%s/golden/input.raw'%(DIR), np.float32).reshape(input_details[0]['shape'])
        output = model.predict({'0':data})
        outR = np.fromfile('%s/golden/output.raw'%(DIR), np.float32)
        compare(outR, output['0'], tfmodel)
    except Exception as e:
        print(e)
