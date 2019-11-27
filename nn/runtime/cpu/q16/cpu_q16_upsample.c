/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q16
#include "../runtime_cpu.h"
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_q16_upsample_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q16_UPSAMPLE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q16_upsample_context_t), sizeof(int16_t));
}

int layer_cpu_q16_UPSAMPLE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q16_upsample_context_t* context = (layer_cpu_q16_upsample_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	uint8_t* pmask = NULL;

	NNLOG(NN_DEBUG, ("execute %s:",layer->name));

	if(2 == input_context->nout)
	{
		pmask = (uint8_t*) input_context->out[1];
	}

	r = alg_up_sampling(context->out[0], input_context->out[0], &context->nhwc, &input_context->nhwc, sizeof(int16_t), pmask);

	return r;
}

void layer_cpu_q16_UPSAMPLE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q16 */
