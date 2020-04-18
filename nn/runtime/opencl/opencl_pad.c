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
} layer_cl_pad_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_PAD_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_create_layer_common(nn, layer,
			OPENCL_PATH "pad.cl", "pad", NULL,
			sizeof(layer_cl_pad_context_t));
}

int layer_cl_PAD_set_args(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_pad_context_t* context = (layer_cl_pad_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;

	int * ints = layer->blobs[0]->blob;

	int padding_top = ints[1];
	int padding_bottom = ints[5];
	int padding_left = ints[2];
	int padding_right = ints[6];

	input_context = (layer_cl_context_t*)input->C->context;

	r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHWC, 6,
				sizeof(cl_mem), &(input_context->out[0]),
				sizeof(cl_mem), &(context->out[0]),
				sizeof(int), &padding_top,
				sizeof(int), &padding_bottom,
				sizeof(int), &padding_left,
				sizeof(int), &padding_right);
	return r;
}
int layer_cl_PAD_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_execute_layer(nn, layer, RTE_GWT_W_H_C, FALSE, NULL);
}

void layer_cl_PAD_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
