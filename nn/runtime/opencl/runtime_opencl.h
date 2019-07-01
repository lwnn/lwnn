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

#define LAYER_CL_CONTEXT_MEMBER		\
	LAYER_CONTEXT_MEMBER;			\
	cl_program program;				\
	cl_kernel kernel;				\
	size_t nout;					\
	cl_mem* out

/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	LAYER_CL_CONTEXT_MEMBER;
} layer_cl_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
#define RTE_CL_NHWC_W(nhwc)		(((nhwc).W)*(((nhwc).C+3)>>2))
#define RTE_CL_NHWC_H(nhwc)		(((nhwc).N)*((nhwc).H))

#define RTE_CL_ARGS_WITH_N	0x01
#define RTE_CL_ARGS_WITH_H	0x02
#define RTE_CL_ARGS_WITH_W	0x04
#define RTE_CL_ARGS_WITH_C	0x08
#define RTE_CL_ARGS_WITH_NHWC	0x0F

#define RTE_CL_LOG_LAYER_SHAPE(layer) 											\
	NNLOG(NN_DEBUG, ("%s dims: [%dx%dx%dx%d] -> [1x%dx%dx4]\n",					\
						layer->name,											\
						layer->C->context->nhwc.N, layer->C->context->nhwc.H,	\
						layer->C->context->nhwc.W, layer->C->context->nhwc.C,	\
						RTE_CL_NHWC_H(layer->C->context->nhwc),					\
						RTE_CL_NHWC_W(layer->C->context->nhwc)))

/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
cl_mem rte_cl_create_buffer(const nn_t* nn, size_t sz, float* init_value);
cl_mem rte_cl_create_image2d(const nn_t* nn, int H, int W);
int rte_cl_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			const char* program, const char* kernel,
			size_t sz, size_t nout);
void rte_cl_destory_layer_context(const nn_t* nn, const layer_t* layer);
int rte_cl_set_layer_args(
			const nn_t* nn, const layer_t* layer,
			uint32_t nhwc, size_t num, ...);
int rte_cl_execute_layer(const nn_t* nn, const layer_t* layer, int use_cl_hw);
int rte_cl_read_buffer(const nn_t* nn, cl_mem buffer, void* data, size_t sz);
#endif /* NN_RUNTIME_OPENCL_RUNTIME_OPENCL_H_ */
