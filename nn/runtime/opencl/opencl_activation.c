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
} layer_cl_activation_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int layer_cl_activation_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	const char* kernel;

	switch(layer->op)
	{
		case L_OP_RELU:
			kernel = "relu";
			break;
		default:
			assert(0);
			break;
	}

	r = rte_cl_create_layer_common(nn, layer,
				OPENCL_PATH "activation.cl", kernel,
				sizeof(layer_cl_activation_context_t));

	return r;
}

static int layer_cl_activation_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_activation_context_t* context = (layer_cl_activation_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;

	input_context = (layer_cl_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	r = rte_cl_set_layer_args(nn, layer, 0, 2,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->out[0]));

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer, RTE_GWT_CL_W_H);
	}

	return r;
}

static void layer_cl_activation_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_RELU_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_init(nn, layer);
}

int layer_cl_RELU_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_execute(nn, layer);
}

void layer_cl_RELU_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_activation_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_OPENCL */
