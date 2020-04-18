/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q16
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q16_CONTEXT_MEMBER;
} layer_cpu_q16_reshape_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q16_RESHAPE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_q16_reshape_context_t), 1);
}

int layer_cpu_q16_RESHAPE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q16_reshape_context_t* context = (layer_cpu_q16_reshape_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q16_context_t* input_context = (layer_cpu_q16_context_t*)input->C->context;

	int16_t* IN = (int16_t*)input_context->out[0];

	context->out[0] = IN;	/* yes, just set up the output */

	return r;
}

void layer_cpu_q16_RESHAPE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q16 */
