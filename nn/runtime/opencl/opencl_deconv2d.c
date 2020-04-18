/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
	cl_mem W;
	cl_mem B;
} layer_cl_deconv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_DECONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	const char* option = NULL;
	layer_cl_deconv2d_context_t* context;
	layer_activation_type_t act = RTE_FETCH_INT32(layer->blobs[2]->blob, 6);

	switch(act) {
		case L_ACT_RELU:
			option = "-DRELU";
			break;
		case L_ACT_LEAKY:
			option = "-DLEAKY";
			break;
		default:
			break;
	}

	r = rte_cl_create_layer_common(nn, layer,
			OPENCL_PATH "deconv2d.cl", "deconv2d", option,
			sizeof(layer_cl_deconv2d_context_t));

	if(0 == r)
	{
		context = (layer_cl_deconv2d_context_t*)layer->C->context;

		context->W = rte_cl_create_image2d_from_blob(nn, layer->blobs[0]);
		context->B = rte_cl_create_image2d_from_blob(nn, layer->blobs[1]);
		if((NULL == context->W) || (NULL == context->B))
		{
			r = NN_E_NO_MEMORY;
		}
	}

	return r;
}
int layer_cl_DECONV2D_set_args(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_deconv2d_context_t* context = (layer_cl_deconv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;
	int knlX, knlY, padX, padY, strideX, strideY;
	int* ints;

	input_context = (layer_cl_context_t*)input->C->context;

	ints = (int*)layer->blobs[0]->dims;
	knlY = ints[1];
	knlX = ints[2];

	ints = (int*)layer->blobs[2]->blob;

	strideY = ints[4];
	strideX = ints[5];

	padY = alg_deconv2d_calculate_padding(knlY, strideY, input_context->nhwc.H, context->nhwc.H);
	padX = alg_deconv2d_calculate_padding(knlX, strideX, input_context->nhwc.W, context->nhwc.W);

	r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHWC, 13,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->W),
					sizeof(cl_mem), &(context->B),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(int), &(input_context->nhwc.W),
					sizeof(int), &(input_context->nhwc.H),
					sizeof(int), &(input_context->nhwc.C),
					sizeof(int), &knlX,
					sizeof(int), &knlY,
					sizeof(int), &padX,
					sizeof(int), &padY,
					sizeof(int), &strideX,
					sizeof(int), &strideY);

	return r;
}

int layer_cl_DECONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_execute_layer(nn, layer, RTE_GWT_W_H_C, FALSE, NULL);
}

void layer_cl_DECONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_deconv2d_context_t* context;

	context = (layer_cl_deconv2d_context_t*)layer->C->context;

	if(NULL != context)
	{
		rte_cl_destory_memory(context->W);
		rte_cl_destory_memory(context->B);
	}

	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
