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
	cl_mem W;
} layer_cl_activation_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int layer_cl_activation_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	const char* kernel;
	layer_cl_activation_context_t* context;

	switch(layer->op)
	{
		case L_OP_RELU:
			kernel = "relu";
			break;
		case L_OP_PRELU:
			kernel = "prelu";
			break;
		case L_OP_CLIP:
			kernel = "clip";
			break;
		default:
			assert(0);
			break;
	}

	r = rte_cl_create_layer_common(nn, layer,
				OPENCL_PATH "activation.cl", kernel, NULL,
				sizeof(layer_cl_activation_context_t));

	if((0 == r) && (L_OP_PRELU == layer->op))
	{
		context = (layer_cl_activation_context_t*)layer->C->context;

		context->W = rte_cl_create_image2d_from_blob(nn, layer->blobs[0]);
		if(NULL == context->W)
		{
			r = NN_E_NO_MEMORY;
		}
	}
	return r;
}

static int layer_cl_activation_set_args(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_activation_context_t* context = (layer_cl_activation_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;

	input_context = (layer_cl_context_t*)input->C->context;

	switch(layer->op) {
		case L_OP_RELU:
			r = rte_cl_set_layer_args(nn, layer, 0, 2,
								sizeof(cl_mem), &(input_context->out[0]),
								sizeof(cl_mem), &(context->out[0]));
			break;
		case L_OP_PRELU: {
			int nC4 = (context->nhwc.C+3)>>2;
			r = rte_cl_set_layer_args(nn, layer, 0, 4,
							sizeof(cl_mem), &(input_context->out[0]),
							sizeof(cl_mem), &(context->W),
							sizeof(cl_mem), &(context->out[0]),
							sizeof(int), &nC4);
			break;
		}
		case L_OP_CLIP: {
			float min = RTE_FETCH_FLOAT(layer->blobs[0]->blob, 0);
			float max = RTE_FETCH_FLOAT(layer->blobs[0]->blob, 1);
			r = rte_cl_set_layer_args(nn, layer, 0, 2,
						sizeof(cl_mem), &(input_context->out[0]),
						sizeof(cl_mem), &(context->out[0]),
						sizeof(float), &min,
						sizeof(float), &max);
			break;
		}
		default:
			break;
	}

	return r;
}
static int layer_cl_activation_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_execute_layer(nn, layer, RTE_GWT_CL_W_H, FALSE, NULL);
}

static void layer_cl_activation_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_activation_context_t* context;

	context = (layer_cl_activation_context_t*)layer->C->context;

	if(NULL != context)
	{
		rte_cl_destory_memory(context->W);
	}

	rte_cl_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_RELU_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_init(nn, layer);
}

int layer_cl_RELU_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_set_args(nn, layer);
}

int layer_cl_RELU_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_execute(nn, layer);
}

void layer_cl_RELU_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_activation_deinit(nn, layer);
}

int layer_cl_PRELU_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_init(nn, layer);
}

int layer_cl_PRELU_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_set_args(nn, layer);
}

int layer_cl_PRELU_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_execute(nn, layer);
}

void layer_cl_PRELU_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_activation_deinit(nn, layer);
}

int layer_cl_CLIP_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_init(nn, layer);
}

int layer_cl_CLIP_set_args(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_set_args(nn, layer);
}

int layer_cl_CLIP_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cl_activation_execute(nn, layer);
}

void layer_cl_CLIP_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_activation_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_OPENCL */
