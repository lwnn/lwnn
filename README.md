# LWNN - Lightweight Neural Network

[![Build Status](https://travis-ci.org/lwnn/lwnn.svg?branch=master)](https://travis-ci.org/lwnn/lwnn)

Mostly, inspired by [NNOM](https://github.com/majianjia/nnom), [CMSIS-NN](https://github.com/ARM-software/CMSIS_5/tree/develop/CMSIS/NN), I want to do something for Edge AI.

But as I think NNOM is not well designed for different runtime, CPU/DSP/GPU/NPU etc, it doesn't have a clear path to handle different type of runtime, and nowdays, I really want to study somehing about OpenCL, and I came across [MACE](https://github.com/XiaoMi/mace/tree/master/mace/ops/opencl/cl), and I find there is a bunch of CL kernels can be used directly.

So I decieded to do something meaningfull, do some study of OpenCL and at the meantime to create a Lightweight Neural Network that can be suitale for decices such as PC, mobiles and MCU etc.

## Architecture

And for the purpose to support variant Deep Learning frameworks such as tensorflow/keras/caffe2, pytorch etc, the [onnx](https://onnx.ai/) will be supported by lwnn.

![arch](docs/arch.png)

| Layers/Runtime | cpu float | cpu s8 | cpu q8 | cpu q16 | opencl | comments |
| - | - | - | - | - | - | - |
| Conv1D | Y | Y | Y | Y | Y | based on Conv2D |
| Conv2D | Y | Y | Y | Y | Y | |
| DepthwiseConv2D | Y | N | Y | Y | Y | |
| EltmentWise Max | Y | N | Y | Y | Y | |
| ReLU | Y | N | Y | Y | Y | |
| MaxPool1D | Y | N | Y | Y | Y | based on MaxPool2D |
| MaxPool2D | Y | N | Y | Y | Y | |
| Dense | Y | N | Y | Y | Y | |
| Softmax | Y | N | Y | Y | Y | |
| Reshape | Y | N | Y | Y | Y | |
| Pad | Y | N | Y | Y | Y | |
| BatchNorm | Y | N | Y | Y | Y | only support BatchNorm after Conv2D |
| Concat | Y | N | Y | Y | Y | |
| AvgPool1D | Y | N | Y | Y | Y | based on AvgPool2D |
| AvgPool2D | Y | N | Y | Y | Y | |
| Add | Y | N | Y | Y | Y | |

## Development

### prepare environment
```sh
conda create -n lwnn python=3.6
source activate lwnn
conda install scons 
pip install tensorflow keras keras2onnx onnxruntime
sudo apt install nvidia-opencl-dev
```

### build

```sh
scons
```
