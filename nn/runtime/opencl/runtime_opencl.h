/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_RUNTIME_OPENCL_RUNTIME_OPENCL_H_
#define NN_RUNTIME_OPENCL_RUNTIME_OPENCL_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include <CL/cl.h>
/* ============================ [ MACROS    ] ====================================================== */
#define OPENCL_ROUNDUP4(c) ((c+3)&(~0x3))
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
cl_mem runtime_opencl_create_image2d(const nn_t* nn, int H, int W);
#endif /* NN_RUNTIME_OPENCL_RUNTIME_OPENCL_H_ */
