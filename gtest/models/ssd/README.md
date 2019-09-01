
from https://github.com/chuanqi305/MobileNet-SSD

generate no_bn model by "python merge_bn.py"

then gen lwnn model by "python lwnn/tools/caffe2lwnn.py -i no_bn.prototxt  -w no_bn.caffemodel -o ssd"

insert below python code to demo.py to generate goldens:

```python
id = 0
def detect(imgfile):
    ...
    ...
    global id
    data = np.array(net.blobs['data'].data, np.float32).transpose(0,2,3,1)
    data.tofile('models/mobilenet_ssd/golden/input%d.raw'%(id))
    o = np.array(out['detection_out'].data, np.float32)
    o.tofile('models/mobilenet_ssd/golden/output%d.raw'%(id))
    id+=1
```

then copy the lwnn source file and its goldens here.
