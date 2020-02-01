
# Caffe SSD

from https://github.com/chuanqi305/MobileNet-SSD

generate no_bn model by "python merge_bn.py"

insert below python code to demo.py to generate goldens:

```python
id = 0
def detect(imgfile):
    ...
    ...
    global id
    data = np.array(net.blobs['data'].data, np.float32).transpose(0,2,3,1)
    data.tofile('models/ssd/golden/input%d.raw'%(id))
    o = np.array(out['detection_out'], np.float32)
    o.tofile('models/ssd/golden/output%d.raw'%(id))
    id+=1
```

then gen lwnn model by "python lwnn/tools/caffe2lwnn.py -i no_bn.prototxt  -w no_bn.caffemodel -o ssd -r models/ssd/golden"


then copy the lwnn source file and its goldens here.

# OpenVINO SSD

from [ssd300-int8-sparse-v2-onnx-0001](https://download.01.org/opencv/2019/open_model_zoo/R4/20191121_190000_models_bin/ssd300-int8-sparse-v2-onnx-0001/FP32-INT8/) download the model


