
from https://github.com/chuanqi305/MobileNet-SSD

generate no_bn model by "python merge_bn.py"

then gen lwnn model by "python lwnn/tools/caffe2lwnn.py -i no_bn.prototxt  -w no_bn.caffemodel -o mobilenet_ssd"

then copy the lwnn source file here.
