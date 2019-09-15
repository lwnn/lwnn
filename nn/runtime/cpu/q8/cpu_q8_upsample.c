/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#if !defined(DISABLE_RUNTIME_CPU_Q8) || !defined(DISABLE_RUNTIME_CPU_S8)
#include "../runtime_cpu.h"
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_q8_upsample_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_UPSAMPLE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_upsample_context_t), sizeof(int8_t));
}

int layer_cpu_q8_UPSAMPLE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_upsample_context_t* context = (layer_cpu_q8_upsample_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s:",layer->name));

	r = alg_up_sampling(context->out[0], input_context->out[0], &context->nhwc, &input_context->nhwc, sizeof(int8_t));

	return r;
}

void layer_cpu_q8_UPSAMPLE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#ifndef DISABLE_RUNTIME_CPU_S8
int layer_cpu_s8_UPSAMPLE_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_UPSAMPLE_init(nn, layer);
}
int layer_cpu_s8_UPSAMPLE_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_UPSAMPLE_execute(nn, layer);
}
void layer_cpu_s8_UPSAMPLE_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_UPSAMPLE_deinit(nn, layer);
}
#endif
#endif /* DISABLE_RUNTIME_CPU_Q8 */
