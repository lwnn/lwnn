/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_actvation_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void relu_ref(float * data, size_t size)
{
	size_t  i;

	for (i = 0; i < size; i++)
	{
		if (data[i] < 0)
			data[i] = 0;
	}
}
static int layer_cpu_float_activation_init(const nn_t* nn, const layer_t* layer)
{
	int r =0;
	layer_cpu_float_actvation_context_t* context;

	const layer_t* input;
	layer_cpu_context_t* input_context;

	r = rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_float_actvation_context_t), 1);

	if(0 == r)
	{
		context = (layer_cpu_float_actvation_context_t*)layer->C->context;

		input = layer->inputs[0];
		input_context = (layer_cpu_context_t*)input->C->context;

		if(NULL != input_context->out[0])
		{
			/* reuse its input layer's output buffer */
			rte_cpu_take_buffer(input_context->out[0], layer);
		}
	}

	return r;
}

static int layer_cpu_float_activation_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_actvation_context_t* context = (layer_cpu_float_actvation_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context;
	size_t sz = NHWC_SIZE(context->nhwc);
	float* IN;

	input_context = (layer_cpu_context_t*)input->C->context;

	IN = (float*)input_context->out[0];

	context->out[0] = IN;	/* yes, reuse its input's output buffer directly */

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	switch(layer->op)
	{
		case L_OP_RELU:
			relu_ref(IN, sz);
			break;
		default:
			r = NN_E_INVALID_LAYER;
			break;
	}

	return r;
}

static void layer_cpu_float_activation_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_RELU_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_activation_init(nn, layer);
}

int layer_cpu_float_RELU_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_activation_execute(nn, layer);
}

void layer_cpu_float_RELU_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_activation_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_FLOAT */
