# LWNN - Lightweight Neural Network
# Copyright (C) 2019  Parai Wang <parai@foxmail.com>

import numpy as np
import matplotlib.pyplot as plt
import math

def show(mts):
    if(mts is None):
        print('>>> nothing to show  <<<')
    if(type(mts) != list):
        mts = [mts]

    for i, mt in enumerate(mts):
        X = mt.shape[0]
        Y = mt.shape[1]
        fig, ax = plt.subplots(figsize=(Y, X))
        im = ax.imshow(mt)

        # We want to show all ticks...
        ax.set_xticks(np.arange(Y))
        ax.set_yticks(np.arange(X))
        # ... and label them with the respective list entries
        ax.set_xticklabels([y for y in range(Y)])
        ax.set_yticklabels([x for x in range(X)])

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        for x in range(X):
            for y in range(Y):
                text = ax.text(y, x, mt[x, y], fontweight='bold',fontsize='x-large',
                               ha="center", va="center", color="w")

        ax.set_title('matrix%d'%(i))
        fig.tight_layout()
    plt.show()

def compare(a, b, name='', delayedShow=False, onlyCompared=False, gn='golden', tn='lwnn'):
    aL = a.reshape(-1).tolist()
    bL = b.reshape(-1).tolist()
    if(len(aL) != len(bL)):
        raise Exception('a len=%s != b len=%s'%(len(aL), len(bL)))
    Z = list(zip(aL,bL))
    Z.sort(key=lambda x: x[0])
    aL1,bL1=zip(*Z)
    if(onlyCompared==False):
        plt.figure(figsize=(18, 3))
        plt.subplot(131)
        plt.plot(aL)
        plt.plot(aL1,'r')
        plt.grid()
        plt.title('%s-%s'%(gn,name))
        plt.subplot(132)
        bL2=list(bL)
        bL2.sort()
        plt.plot(bL)
        plt.plot(bL2,'g')
        plt.grid()
        plt.title('%s-%s'%(tn,name))
        plt.subplot(133)
    else:
        plt.figure()
    plt.plot(bL1,'g')
    plt.plot(aL1,'r')
    plt.grid()
    if(onlyCompared==False):
        plt.title('compare')
    else:
        plt.title(name)
    if(not delayedShow):
        plt.show()

if(__name__ == '__main__'):
    import argparse
    parser = argparse.ArgumentParser(description='verify output with golden')
    parser.add_argument('-i', '--input', help='lwnn output', type=str, required=True)
    parser.add_argument('-g', '--golden', help='golden output', type=str, required=True)
    parser.add_argument('-t', '--type', help='type: one of float, s8, q8, q16', type=str, default='float', required=False)
    parser.add_argument('-Q', '--Q', help='number of Q', type=int, default=None, required=False)
    parser.add_argument('-S', '--S', help='value of Q Scale', type=int, default=None, required=False)
    parser.add_argument('-Z', '--Z', help='value of Q Zero Point', type=int, default=None, required=False)
    parser.add_argument('-o', '--only_compared', help='show only compared result', default=False, action='store_true', required=False)
    args = parser.parse_args()

    if((args.type != 'float') and (args.Q == None)):
        print('please give Q parameter')
        parser.print_help()
        exit()

    if((args.type == 's8') and ((args.S == None) or (args.Z == None))):
        print('please give Scale/Zero Point parameter')
        parser.print_help()
        exit()

    if(args.type == 'float'):
        inp = np.fromfile(args.input, dtype=np.float32)
    elif(args.type == 's8'):
        inp = np.fromfile(args.input, dtype=np.int8).astype(np.float32)
        inp = (args.S/(1<<16))*(inp+args.Z)*math.pow(2, -args.Q)
    elif(args.type == 'q8'):
        inp = np.fromfile(args.input, dtype=np.int8)*math.pow(2, -args.Q)
    elif(args.type == 'q16'):
        inp = np.fromfile(args.input, dtype=np.int16)*math.pow(2, -args.Q)

    golden = np.fromfile(args.golden, dtype=np.float32)

    if(inp.shape != golden.shape):
        if(args.type == 'float'):
            golden = np.fromfile(args.golden, dtype=np.float32)
        elif(args.type == 's8'):
            golden = np.fromfile(args.golden, dtype=np.int8).astype(np.float32)
            golden = (args.S/(1<<16))*(golden+args.Z)*math.pow(2, -args.Q)
        elif(args.type == 'q8'):
            golden = np.fromfile(args.golden, dtype=np.int8)*math.pow(2, -args.Q)
        elif(args.type == 'q16'):
            golden = np.fromfile(args.golden, dtype=np.int16)*math.pow(2, -args.Q)

    if(inp.shape != golden.shape):
        print('shape mismatch: input=%s golden=%s\n'
              'please give type or maybe incorrect input!'%(
                  inp.shape, golden.shape))
        parser.print_help()
        exit()

    compare(golden, inp, onlyCompared=args.only_compared)


