/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
} layer_cl_pooling_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int layer_cl_pooling_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int with_mask = FALSE;
	int nout = 1;
	const char* kernel;
	const char* option = NULL;
	layer_cl_pooling_context_t* context;

	switch(layer->op)
	{
		case L_OP_MAXPOOL:
			kernel = "maxpool";
			with_mask = RTE_FETCH_INT32(layer->blobs[0]->blob, 6);
			if(with_mask) {
				option = "-DWITH_MASK";
				nout = 2;
			}
			break;
		case L_OP_AVGPOOL:
			kernel = "avgpool";
			break;
		default:
			assert(0);
			break;
	}

	r = rte_cl_create_layer_context(nn, layer,
				OPENCL_PATH "pooling.cl", kernel, option,
				sizeof(layer_cl_pooling_context_t), nout);

	if(0 == r) {
		context = (layer_cl_pooling_context_t*)layer->C->context;

		RTE_CL_LOG_LAYER_SHAPE(layer);
		#ifdef ENABLE_CL_IMAGE_REUSE
		context->out[0] = (cl_mem)rte_cl_alloc_image2d(nn, layer,
		#else
		context->out[0] = (cl_mem)rte_cl_create_image2d(nn,
		#endif
					RTE_CL_NHWC_H(context->nhwc),
					RTE_CL_NHWC_W(context->nhwc),
					CL_FLOAT);

		if(NULL == context->out[0])
		{
			r = NN_E_NO_MEMORY;
			rte_cl_destory_layer_context(nn, layer);
		}
	}

	if((0 == r) && (with_mask))
	{
		#ifdef ENABLE_CL_IMAGE_REUSE
		context->out[1] = (cl_mem)rte_cl_alloc_image2d(nn, layer,
		#else
		context->out[1] = (cl_mem)rte_cl_create_image2d(nn,
		#endif
					RTE_CL_NHWC_H(context->nhwc),
					RTE_CL_NHWC_W(context->nhwc),
					CL_UNSIGNED_INT32);

		if(NULL == context->out[1])
		{
			r = NN_E_NO_MEMORY;
			rte_cl_destory_layer_context(nn, layer);
		}
	}

	return r;
}

static int layer_cl_pooling_set_args(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_pooling_context_t* context = (layer_cl_pooling_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;
	int knlX, knlY, padX, padY, strideX, strideY;
	int* ints;
	int with_mask;

	input_context = (layer_cl_context_t*)input->C->context;

	ints = (int*)layer->blobs[0]->blob;
	knlY = ints[0];
	knlX = ints[1];
	padY = ints[2];
	padX = ints[3];
	strideY = ints[4];
	strideX = ints[5];
	with_mask = ints[6];

	if(with_mask) {
		r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHC, 11,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(cl_mem), &(context->out[1]),
					sizeof(int), &(input_context->nhwc.W),
					sizeof(int), &(input_context->nhwc.H),
					sizeof(int), &knlX,
					sizeof(int), &knlY,
					sizeof(int), &padX,
					sizeof(int), &padY,
					sizeof(int), &strideX,
					sizeof(int), &strideY);
	} else {
		r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHC, 10,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(int), &(input_context->nhwc.W),
					sizeof(int), &(input_context->nhwc.H),
					sizeof(int), &knlX,
					sizeof(int), &knlY,
					sizeof(int), &padX,
					sizeof(int), &padY,
					sizeof(int), &strideX,
					sizeof(int), &strideY);
	}
	return r;
}

static int layer_cl_pooling_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_execute_layer(nn, layer, RTE_GWT_W_H_C, FALSE, NULL);
}

static void layer_cl_pooling_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_MAXPOOL_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_pooling_init(nn, layer);
}

int layer_cl_MAXPOOL_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_pooling_set_args(nn, layer);
}

int layer_cl_MAXPOOL_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_pooling_execute(nn, layer);
}

void layer_cl_MAXPOOL_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_pooling_deinit(nn, layer);
}

int layer_cl_AVGPOOL_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_pooling_init(nn, layer);
}

int layer_cl_AVGPOOL_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_pooling_set_args(nn, layer);
}

int layer_cl_AVGPOOL_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_pooling_execute(nn, layer);
}

void layer_cl_AVGPOOL_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_pooling_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_OPENCL */
