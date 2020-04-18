/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q8
#include "../runtime_cpu.h"

#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
} layer_cpu_q8_actvation_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void clip_q8_ref(int8_t* data, size_t size, int8_t min, int8_t max)
{
	size_t  i;

	for (i = 0; i < size; i++)
	{
		if (data[i] < min) {
			data[i] = min;
		}
		else if (data[i] > max) {
			data[i] = max;
		} else {
			/* pass */
		}
	}
}

static int layer_cpu_q8_activation_init(const nn_t* nn, const layer_t* layer)
{
	int r =0;
	const layer_t* input;
	layer_cpu_q8_context_t* input_context;

	r = rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_q8_actvation_context_t), 1);

	if(0 == r)
	{
		input = layer->inputs[0];
		input_context = (layer_cpu_q8_context_t*)input->C->context;

		if(NULL != input_context->out[0])
		{
			/* reuse its input layer's output buffer */
			rte_cpu_take_buffer(input_context->out[0], layer, 0);
		}
	}

	return r;
}

static int layer_cpu_q8_activation_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_actvation_context_t* context = (layer_cpu_q8_actvation_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q8_context_t* input_context;
	size_t sz = NHWC_SIZE(context->nhwc);
	int8_t* IN;

	input_context = (layer_cpu_q8_context_t*)input->C->context;

	IN = (int8_t*)input_context->out[0];

	context->out[0] = IN;	/* yes, reuse its input's output buffer directly */

	switch(layer->op)
	{
		case L_OP_RELU:
			arm_relu_q7(IN, sz);
			break;
		case L_OP_CLIP:
		{
			int8_t min = RTE_FETCH_INT8(layer->blobs[1]->blob, 0);
			int8_t max = RTE_FETCH_INT8(layer->blobs[1]->blob, 1);
			clip_q8_ref(IN, sz, min, max);
			break;
		}
		default:
			r = NN_E_INVALID_LAYER;
			break;
	}

	return r;
}

static void layer_cpu_q8_activation_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_RELU_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_activation_init(nn, layer);
}

int layer_cpu_q8_RELU_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_activation_execute(nn, layer);
}

void layer_cpu_q8_RELU_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_activation_deinit(nn, layer);
}

int layer_cpu_q8_CLIP_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_activation_init(nn, layer);
}

int layer_cpu_q8_CLIP_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_activation_execute(nn, layer);
}

void layer_cpu_q8_CLIP_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_activation_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_Q8 */
