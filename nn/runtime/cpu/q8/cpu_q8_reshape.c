/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#if !defined(DISABLE_RUNTIME_CPU_Q8) || !defined(DISABLE_RUNTIME_CPU_S8)
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
} layer_cpu_q8_reshape_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_RESHAPE_init(const nn_t* nn, const layer_t* layer)
{
	int r =0;
	const layer_t* input;
	layer_cpu_q8_context_t* input_context;

	r = rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_q8_reshape_context_t), 1);

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

int layer_cpu_q8_RESHAPE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_reshape_context_t* context = (layer_cpu_q8_reshape_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q8_context_t* input_context = (layer_cpu_q8_context_t*)input->C->context;

	int8_t* IN = (int8_t*)input_context->out[0];

	context->out[0] = IN;	/* yes, just set up the output */

	return r;
}

void layer_cpu_q8_RESHAPE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#ifndef DISABLE_RUNTIME_CPU_S8
int layer_cpu_s8_RESHAPE_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_RESHAPE_init(nn, layer);
}
int layer_cpu_s8_RESHAPE_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_RESHAPE_execute(nn, layer);
}
void layer_cpu_s8_RESHAPE_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_RESHAPE_deinit(nn, layer);
}
#endif
#endif /* DISABLE_RUNTIME_CPU_Q8 */
