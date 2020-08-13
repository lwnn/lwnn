/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
#define DEFAULT_CL_SET_ARGS(op) \
	int layer_cl_##op##_set_args(const nn_t* nn, const layer_t* layer) { return 0; }
#define UNSUPPORTED_LAYER_OPS_CL(op)	\
		UNSUPPORTED_LAYER_OPS(cl, op)	\
		DEFAULT_CL_SET_ARGS(op)

#ifndef DISABLE_RUNTIME_CPU_Q8
#define FALLBACK_LAYER_OPS_CPU_Q8(op, to) FALLBACK_LAYER_OPS(cpu_q8, op, to)
#else
#define FALLBACK_LAYER_OPS_CPU_Q8(op, to)
#endif
#ifndef DISABLE_RUNTIME_CPU_Q16
#define FALLBACK_LAYER_OPS_CPU_Q16(op, to) FALLBACK_LAYER_OPS(cpu_q16, op, to)
#else
#define FALLBACK_LAYER_OPS_CPU_Q16(op, to)
#endif
#ifndef DISABLE_RUNTIME_CPU_S8
#define FALLBACK_LAYER_OPS_CPU_S8(op, to) FALLBACK_LAYER_OPS(cpu_s8, op, to)
#else
#define FALLBACK_LAYER_OPS_CPU_S8(op, to)
#endif
#ifndef DISABLE_RUNTIME_OPENCL
#define FALLBACK_LAYER_OPS_CL(op, to) FALLBACK_LAYER_OPS(cl, op, to) DEFAULT_CL_SET_ARGS(op)
#else
#define FALLBACK_LAYER_OPS_CL(op, to)
#endif

#ifndef DISABLE_RTE_FALLBACK
#define FALLBACK_TEMPLATE(from, to, gf)													\
extern void rte_##gf##_to_##to##_init_common(const nn_t*, const layer_t*);				\
extern int rte_##gf##_to_##to##_pre_execute_common(const nn_t*, const layer_t*);		\
extern void rte_##gf##_to_##to##_post_execute_common(const nn_t*, const layer_t*);		\
extern void rte_##gf##_to_##to##_deinit_common(const nn_t*, const layer_t*);			\
void layer_##from##_to_##to##_init_common(const nn_t* nn, const layer_t* layer)			\
{																						\
	rte_##gf##_to_##to##_init_common(nn, layer);										\
}																						\
int layer_##from##_to_##to##_pre_execute_common(const nn_t* nn, const layer_t* layer)	\
{																						\
	return rte_##gf##_to_##to##_pre_execute_common(nn, layer);							\
}																						\
void layer_##from##_to_##to##_post_execute_common(const nn_t* nn, const layer_t* layer)	\
{																						\
	rte_##gf##_to_##to##_post_execute_common(nn, layer);								\
}																						\
void layer_##from##_to_##to##_deinit_common(const nn_t* nn, const layer_t* layer)		\
{																						\
	rte_##gf##_to_##to##_deinit_common(nn, layer);										\
}
#else
#define FALLBACK_TEMPLATE(from, to, gf)
#endif

#ifndef DISABLE_RUNTIME_CPU_Q8
#define FALLBACK_RTE_CPU_Q8(op, to) FALLBACK_TEMPLATE(cpu_q8, op, to)
#else
#define FALLBACK_RTE_CPU_Q8(op, to)
#endif
#ifndef DISABLE_RUNTIME_CPU_Q16
#define FALLBACK_RTE_CPU_Q16(op, to) FALLBACK_TEMPLATE(cpu_q16, op, to)
#else
#define FALLBACK_RTE_CPU_Q16(op, to)
#endif
#ifndef DISABLE_RUNTIME_CPU_S8
#define FALLBACK_RTE_CPU_S8(op, to) FALLBACK_TEMPLATE(cpu_s8, op, to)
#else
#define FALLBACK_RTE_CPU_S8(op, to)
#endif
#ifndef DISABLE_RUNTIME_OPENCL
#define FALLBACK_RTE_CL(op, to) FALLBACK_TEMPLATE(cl, op, to)
#else
#define FALLBACK_RTE_CL(op, to)
#endif
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_get_blob_NHWC(const layer_blob_t* blob, NHWC_t* nhwc)
{
	int r = 0;
	int dim = 0;

	const int* dims = blob->dims;

	assert(dims != NULL);

	while((dims[dim] != 0) && (dim < 4)) {
		dim ++;
	};

	switch(dim)
	{
		case 1:
			nhwc->N = 1;
			nhwc->H = 1;
			nhwc->W = 1;
			nhwc->C = dims[0];
			break;
		case 2:
			nhwc->N = 1;
			nhwc->H = dims[0];
			nhwc->W = 1;
			nhwc->C = dims[1];
			break;
		case 3:
			nhwc->N = 1;
			nhwc->H = dims[0];
			nhwc->W = dims[1];
			nhwc->C = dims[2];
			break;
		case 4:
			nhwc->N = dims[0];
			nhwc->H = dims[1];
			nhwc->W = dims[2];
			nhwc->C = dims[3];
			break;
		default:
			NNLOG(NN_ERROR, ("invalid dimension of blob\n"));
			r = NN_E_INVALID_DIMENSION;
			break;
	}

	return r;
}

int layer_get_NHWC(const layer_t* layer, NHWC_t* nhwc)
{
	int r = 0;
	int dim = 0;
	const int* dims = layer->dims;

	assert(dims != NULL);

	while((dims[dim] != 0) && (dim < 4)) {
		dim ++;
	};

	switch(dim)
	{
		case 1:
			nhwc->N = 1;
			nhwc->H = 1;
			nhwc->W = 1;
			nhwc->C = dims[0];
			break;
		case 2:
			nhwc->N = dims[0];
			nhwc->H = 1;
			nhwc->W = 1;
			nhwc->C = dims[1];
			break;
		case 3:
			nhwc->N = dims[0];
			nhwc->H = dims[1];
			nhwc->W = 1;
			nhwc->C = dims[2];
			break;
		case 4:
			nhwc->N = dims[0];
			nhwc->H = dims[1];
			nhwc->W = dims[2];
			nhwc->C = dims[3];
			break;
		default:
			NNLOG(NN_ERROR, ("invalid dimension for %s\n", layer->name));
			r = NN_E_INVALID_DIMENSION;
			break;
	}

	#ifndef DISABLE_DYNAMIC_SHAPE
	if((-1==nhwc->H) && (-1==nhwc->W)) {
		nhwc->W = 1;
	}
	dim = 0;
	if(-1==nhwc->N) dim++;
	if(-1==nhwc->H) dim++;
	if(-1==nhwc->W) dim++;
	if(-1==nhwc->C) dim++;
	if(dim > 1) {
		r = NN_E_INVALID_DIMENSION;
	}
	#endif

	return r;
}

size_t layer_get_size(const layer_t* layer)
{
	int dim = 0;
	size_t sz = 1;

	if(NULL != layer->C->context)
	{
		dim = 4;
		sz = NHWC_SIZE(layer->C->context->nhwc);
	}
	else if(NULL != layer->dims)
	{
		while(layer->dims[dim] != 0) {
			sz *= layer->dims[dim];
			dim ++;
		};
	}
	else
	{
		NNLOG(NN_ERROR, ("can't get %s layer size for now\n", layer->name));
	}

	if(0 == dim)
	{
		sz = 0;
	}

	return sz;
}

#ifndef DISABLE_DYNAMIC_SHAPE
int layer_get_dynamic_axis(const layer_t* layer)
{
	int dim = 0;
	int axis = -1;
	const int* dims = layer->dims;

	assert(dims != NULL);

	while((dims[dim] != 0) && (dim < 4)) {
		if(-1 == dims[dim]) {
			axis = dim;
		}
		dim ++;
	};

	if(1 == dim) {
		/* pass */
	} else {
		assert(dim >= 2);
		if(axis > (dim-2)) {
			axis = 3;
		}
	}

	return axis;
}

void layer_set_dynamic_shape(const layer_t* layer, int axis, size_t total)
{
	size_t bs;
	int* dims = (int*)&(layer->C->context->nhwc);
	dims[axis] = 1;

	bs = NHWC_SIZE(layer->C->context->nhwc);
	assert(bs > 0);
	assert((total%bs) == 0);
	dims[axis] = total/bs;
}
#endif
/* ============================ [ UNSUPPORTED/FALLBACK ] =========================================== */
UNSUPPORTED_LAYER_OPS(cpu_s8, CONST)
UNSUPPORTED_LAYER_OPS(cpu_q8, CONST)
UNSUPPORTED_LAYER_OPS(cpu_q16, CONST)

UNSUPPORTED_LAYER_OPS(cpu_s8, DILCONV2D)
UNSUPPORTED_LAYER_OPS(cpu_q8, DILCONV2D)
UNSUPPORTED_LAYER_OPS(cpu_q16, DILCONV2D)

UNSUPPORTED_LAYER_OPS(cpu_s8, PRELU)
UNSUPPORTED_LAYER_OPS(cpu_q8, PRELU)
UNSUPPORTED_LAYER_OPS(cpu_q16, PRELU)

FALLBACK_LAYER_OPS_CPU_S8(MFCC, cpu_float)
FALLBACK_LAYER_OPS_CPU_Q8(MFCC, cpu_float)
FALLBACK_LAYER_OPS_CPU_Q16(MFCC, cpu_float)
FALLBACK_LAYER_OPS_CL(MFCC, cpu_float)

FALLBACK_LAYER_OPS_CL(TRANSPOSE, cpu_float)

UNSUPPORTED_LAYER_OPS(cpu_s8, LSTM)
UNSUPPORTED_LAYER_OPS(cpu_q16, LSTM)
FALLBACK_LAYER_OPS_CL(LSTM, cpu_float)

UNSUPPORTED_LAYER_OPS(cpu_s8, SLICE)
UNSUPPORTED_LAYER_OPS(cpu_q8, SLICE)
UNSUPPORTED_LAYER_OPS(cpu_q16, SLICE)
UNSUPPORTED_LAYER_OPS_CL(SLICE)

UNSUPPORTED_LAYER_OPS(cpu_s8, DETECTION)
UNSUPPORTED_LAYER_OPS(cpu_q8, DETECTION)
UNSUPPORTED_LAYER_OPS(cpu_q16, DETECTION)
UNSUPPORTED_LAYER_OPS_CL(DETECTION)

UNSUPPORTED_LAYER_OPS(cpu_s8, PROPOSAL)
UNSUPPORTED_LAYER_OPS(cpu_q8, PROPOSAL)
UNSUPPORTED_LAYER_OPS(cpu_q16, PROPOSAL)
UNSUPPORTED_LAYER_OPS_CL(PROPOSAL)

UNSUPPORTED_LAYER_OPS(cpu_s8, PYRAMID_ROI_ALIGN)
UNSUPPORTED_LAYER_OPS(cpu_q8, PYRAMID_ROI_ALIGN)
UNSUPPORTED_LAYER_OPS(cpu_q16, PYRAMID_ROI_ALIGN)
UNSUPPORTED_LAYER_OPS_CL(PYRAMID_ROI_ALIGN)

FALLBACK_LAYER_OPS_CPU_S8(DETECTIONOUTPUT, cpu_float)
FALLBACK_LAYER_OPS_CPU_Q8(DETECTIONOUTPUT, cpu_float)
FALLBACK_LAYER_OPS_CPU_Q16(DETECTIONOUTPUT, cpu_float)
FALLBACK_LAYER_OPS_CL(DETECTIONOUTPUT, cpu_float)

FALLBACK_LAYER_OPS_CPU_S8(YOLO, cpu_float)
FALLBACK_LAYER_OPS_CPU_Q8(YOLO, cpu_float)
FALLBACK_LAYER_OPS_CPU_Q16(YOLO, cpu_float)
FALLBACK_LAYER_OPS_CL(YOLO, cpu_float)

FALLBACK_LAYER_OPS_CPU_S8(YOLOOUTPUT, cpu_float)
FALLBACK_LAYER_OPS_CPU_Q8(YOLOOUTPUT, cpu_float)
FALLBACK_LAYER_OPS_CPU_Q16(YOLOOUTPUT, cpu_float)
FALLBACK_LAYER_OPS_CL(YOLOOUTPUT, cpu_float)

UNSUPPORTED_LAYER_OPS(cpu_s8, MUL)
UNSUPPORTED_LAYER_OPS(cpu_q8, MUL)
UNSUPPORTED_LAYER_OPS(cpu_q16, MUL)

UNSUPPORTED_LAYER_OPS(cpu_s8, NORMALIZE)
UNSUPPORTED_LAYER_OPS(cpu_q8, NORMALIZE)
UNSUPPORTED_LAYER_OPS(cpu_q16, NORMALIZE)
UNSUPPORTED_LAYER_OPS_CL(NORMALIZE)

UNSUPPORTED_LAYER_OPS(cpu_s8, RESIZE)
UNSUPPORTED_LAYER_OPS(cpu_q8, RESIZE)
UNSUPPORTED_LAYER_OPS(cpu_q16, RESIZE)
UNSUPPORTED_LAYER_OPS_CL(RESIZE)

UNSUPPORTED_LAYER_OPS(cpu_s8, LAYER_NORM)
UNSUPPORTED_LAYER_OPS(cpu_q8, LAYER_NORM)
UNSUPPORTED_LAYER_OPS(cpu_q16, LAYER_NORM)
UNSUPPORTED_LAYER_OPS_CL(LAYER_NORM)

UNSUPPORTED_LAYER_OPS(cpu_s8, GATHER)
UNSUPPORTED_LAYER_OPS(cpu_q8, GATHER)
UNSUPPORTED_LAYER_OPS(cpu_q16, GATHER)
UNSUPPORTED_LAYER_OPS_CL(GATHER)

UNSUPPORTED_LAYER_OPS(cpu_s8, SUB)
UNSUPPORTED_LAYER_OPS(cpu_q8, SUB)
UNSUPPORTED_LAYER_OPS(cpu_q16, SUB)
UNSUPPORTED_LAYER_OPS_CL(SUB)

UNSUPPORTED_LAYER_OPS(cpu_s8, STRIDEDSLICE)
UNSUPPORTED_LAYER_OPS(cpu_q8, STRIDEDSLICE)
UNSUPPORTED_LAYER_OPS(cpu_q16, STRIDEDSLICE)
UNSUPPORTED_LAYER_OPS_CL(STRIDEDSLICE)

UNSUPPORTED_LAYER_OPS(cpu_s8, POW)
UNSUPPORTED_LAYER_OPS(cpu_q8, POW)
UNSUPPORTED_LAYER_OPS(cpu_q16, POW)
UNSUPPORTED_LAYER_OPS_CL(POW)

UNSUPPORTED_LAYER_OPS(cpu_s8, BATCHMATMUL)
UNSUPPORTED_LAYER_OPS(cpu_q8, BATCHMATMUL)
UNSUPPORTED_LAYER_OPS(cpu_q16, BATCHMATMUL)
UNSUPPORTED_LAYER_OPS_CL(BATCHMATMUL)

UNSUPPORTED_LAYER_OPS(cpu_s8, TANH)
UNSUPPORTED_LAYER_OPS(cpu_q8, TANH)
UNSUPPORTED_LAYER_OPS(cpu_q16, TANH)
UNSUPPORTED_LAYER_OPS_CL(TANH)

UNSUPPORTED_LAYER_OPS(cpu_s8, REDUCEMEAN)
UNSUPPORTED_LAYER_OPS(cpu_q8, REDUCEMEAN)
UNSUPPORTED_LAYER_OPS(cpu_q16, REDUCEMEAN)
UNSUPPORTED_LAYER_OPS_CL(REDUCEMEAN)

UNSUPPORTED_LAYER_OPS(cpu_s8, DIV)
UNSUPPORTED_LAYER_OPS(cpu_q8, DIV)
UNSUPPORTED_LAYER_OPS(cpu_q16, DIV)
UNSUPPORTED_LAYER_OPS_CL(DIV)

UNSUPPORTED_LAYER_OPS(cpu_s8, SQRT)
UNSUPPORTED_LAYER_OPS(cpu_q8, SQRT)
UNSUPPORTED_LAYER_OPS(cpu_q16, SQRT)
UNSUPPORTED_LAYER_OPS_CL(SQRT)

FALLBACK_RTE_CPU_S8(cpu_float, cpuq)
FALLBACK_RTE_CPU_Q8(cpu_float, cpuq)
FALLBACK_RTE_CPU_Q16(cpu_float, cpuq)
FALLBACK_RTE_CL(cpu_float, cl)

