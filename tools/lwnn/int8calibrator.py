# LWNN - Lightweight Neural Network
# Copyright (C) 2020  Parai Wang <parai@foxmail.com>
import numpy as np
import scipy

# https://docs.nvidia.com/deeplearning/sdk/tensorrt-sample-support-guide/index.html#int8_sample
# http://on-demand.gputechconf.com/gtc/2017/presentation/s7310-8-bit-inference-with-tensorrt.pdf
# http://docs.openvinotoolkit.org/2018_R5/_samples_calibration_tool_README.html
# https://docs.openvinotoolkit.org/latest/_inference_engine_tools_calibration_tool_README.html

__all__ = ['KL_divergence']

def KL_divergence(r, q):
    return scipy.stats.entropy(r, q)

def quantize(input):
    return input

def Int8Calibration(intut):
    bin = np.histogram(input, bins=2048)
    for i in range(128 , 2048):
        reference_distribution_P = bin[0:i]
        outliers_count = sum(bin[i:])
        reference_distribution_P[i-1] += outliers_count
        P = reference_distribution_P/sum(reference_distribution_P)
        candidate_distribution_Q = quantize(bin[0:i])