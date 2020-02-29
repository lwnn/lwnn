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
} layer_cl_eltwise_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int layer_cl_eltwise_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	const char* kernel;

	switch(layer->op)
	{
		case L_OP_MAXIMUM:
			kernel = "maximum";
			break;
		case L_OP_ADD:
			kernel = "add";
			break;
		case L_OP_MINIMUM:
			kernel = "minimum";
			break;
		default:
			assert(0);
			break;
	}

	r = rte_cl_create_layer_common(nn, layer,
				OPENCL_PATH "eltwise.cl", kernel, NULL,
				sizeof(layer_cl_eltwise_context_t));

	return r;
}

static int layer_cl_eltwise_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_eltwise_context_t* context = (layer_cl_eltwise_context_t*)layer->C->context;
	const layer_t* inputA = layer->inputs[0];
	const layer_t* inputB = layer->inputs[1];
	layer_cl_context_t* inputA_context;
	layer_cl_context_t* inputB_context;

	inputA_context = (layer_cl_context_t*)inputA->C->context;
	inputB_context = (layer_cl_context_t*)inputB->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	r = rte_cl_set_layer_args(nn, layer, 0, 3,
					sizeof(cl_mem), &(inputA_context->out[0]),
					sizeof(cl_mem), &(inputB_context->out[0]),
					sizeof(cl_mem), &(context->out[0]));

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer, RTE_GWT_CL_W_H, FALSE, NULL);
	}

	return r;
}

static void layer_cl_eltwise_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_MAXIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_init(nn, layer);
}

int layer_cl_MAXIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_execute(nn, layer);
}

void layer_cl_MAXIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_eltwise_deinit(nn, layer);
}

int layer_cl_MINIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_init(nn, layer);
}

int layer_cl_MINIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_execute(nn, layer);
}

void layer_cl_MINIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_eltwise_deinit(nn, layer);
}

int layer_cl_ADD_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_init(nn, layer);
}

int layer_cl_ADD_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_eltwise_execute(nn, layer);
}

void layer_cl_ADD_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_eltwise_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_OPENCL */
