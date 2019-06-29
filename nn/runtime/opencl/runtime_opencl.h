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

#ifndef OPENCL_PATH
#define OPENCL_PATH "nn/runtime/opencl/kernels/"
#endif
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	NHWC_t nhwc;
	cl_program program;
	cl_kernel kernel;
} layer_cl_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
cl_mem runtime_opencl_create_image2d(const nn_t* nn, int H, int W);
void* runtime_opencl_create_context(
			const nn_t* nn, const layer_t* layer,
			const char* program, const char* kernel,
			size_t sz, int *r);
#endif /* NN_RUNTIME_OPENCL_RUNTIME_OPENCL_H_ */
