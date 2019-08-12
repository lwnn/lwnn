# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import numpy as np
import matplotlib.pyplot as plt
import math

def compare(a, b, name=''):
    aL = a.tolist()
    bL = b.tolist()
    assert(len(aL) == len(bL))
    Z = list(zip(aL,bL))
    Z.sort(key=lambda x: x[0])
    aL1,bL1=zip(*Z)
    plt.figure(figsize=(18, 3))
    plt.subplot(131)
    plt.plot(aL)
    plt.plot(aL1,'r')
    plt.grid()
    plt.title('gloden-%s'%(name))
    plt.subplot(133)
    plt.plot(bL1,'g')
    plt.plot(aL1,'r')
    plt.grid()
    plt.title('compare')
    plt.subplot(132)
    bL1=list(bL1)
    bL1.sort()
    plt.plot(bL)
    plt.plot(bL1,'g')
    plt.grid()
    plt.title('lwnn-%s'%(name))
    plt.show()

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='verify output with golden')
    parser.add_argument('-i', '--input', help='lwnn output', type=str, required=True)
    parser.add_argument('-g', '--golden', help='golden output', type=str, required=True)
    parser.add_argument('-t', '--type', help='type: one of float, q8, q16', type=str, default='float', required=False)
    parser.add_argument('-Q', '--Q', help='number of Q', type=int, default=None, required=False)
    args = parser.parse_args()

    if((args.type != 'float') and (args.Q == None)):
        print('please give Q parameter')
        parser.print_help()
        exit()

    if(args.type == 'float'):
        inp = np.fromfile(args.input, dtype=np.float32)
    elif(args.type == 'q8'):
        print('xxx')
        inp = np.fromfile(args.input, dtype=np.int8)*math.pow(2, -args.Q)
    elif(args.type == 'q16'):
        inp = np.fromfile(args.input, dtype=np.int16)*math.pow(2, -args.Q)

    golden = np.fromfile(args.golden, dtype=np.float32)

    if(inp.shape != golden.shape):
        print('shape mismatch: input=%s golden=%s\n'
              'please give type or maybe incorrect input!'%(
                  inp.shape, golden.shape))
        parser.print_help()
        exit()

    compare(golden, inp)


