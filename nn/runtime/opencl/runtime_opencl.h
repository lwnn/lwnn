/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_RUNTIME_OPENCL_RUNTIME_OPENCL_H_
#define NN_RUNTIME_OPENCL_RUNTIME_OPENCL_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include <CL/cl.h>
#ifdef __cplusplus
extern "C" {
#endif
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

typedef enum
{
	RTE_GWT_W_H,
	RTE_GWT_CL_W_H,
	RTE_GWT_W_H_C,
} rte_cl_global_work_type_t;
/* ============================ [ DECLARES  ] ====================================================== */
#define RTE_CL_NHWC_W(nhwc)		(((nhwc).W)*(((nhwc).C+3)>>2))
#define RTE_CL_NHWC_H(nhwc)		(((nhwc).N)*((nhwc).H))
#define RTE_CL_NHWC_C(nhwc)		(((nhwc).C+3)>>2)

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
cl_mem rte_cl_create_buffer(const nn_t* nn, size_t sz, const float* init_value);
cl_mem rte_cl_create_image2d(const nn_t* nn, int H, int W);
int rte_cl_image2d_copy_in(const nn_t* nn, cl_mem img2d, const float* in, NHWC_t* nhwc);
int rte_cl_image2d_copy_out(const nn_t* nn, cl_mem img2d, float* out, NHWC_t* nhwc);
cl_mem rte_cl_create_image2d_from_blob(const nn_t* nn, const layer_blob_t* blob);
void rte_cl_destory_memory(cl_mem mem);
int rte_cl_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			const char* program, const char* kernel,
			size_t sz, size_t nout);
void rte_cl_destory_layer_context(const nn_t* nn, const layer_t* layer);
int rte_cl_set_layer_args(
			const nn_t* nn, const layer_t* layer,
			uint32_t nhwc, size_t num, ...);
int rte_cl_execute_layer(const nn_t* nn, const layer_t* layer, rte_cl_global_work_type_t gwt);
int rte_cl_read_buffer(const nn_t* nn, cl_mem buffer, void* data, size_t sz);

int rte_cl_create_layer_common(const nn_t* nn, const layer_t* layer,
		const char* program, const char* kernel, size_t ctx_sz);
#ifdef __cplusplus
}
#endif
#endif /* NN_RUNTIME_OPENCL_RUNTIME_OPENCL_H_ */
