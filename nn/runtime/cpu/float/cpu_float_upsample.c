/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_upsample_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_UPSAMPLE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_upsample_context_t), sizeof(float));
}

int layer_cpu_float_UPSAMPLE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_upsample_context_t* context = (layer_cpu_float_upsample_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	uint8_t* pmask = NULL;

	if(NULL != layer->inputs[1])
	{
		pmask = (uint8_t*) layer->inputs[1]->C->context->out[1];
	}

	NNLOG(NN_DEBUG, ("execute %s%s:",layer->name, pmask?" with mask":""));

	r = alg_up_sampling(context->out[0], input_context->out[0], &context->nhwc, &input_context->nhwc, sizeof(float), pmask);

	return r;
}

void layer_cpu_float_UPSAMPLE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
