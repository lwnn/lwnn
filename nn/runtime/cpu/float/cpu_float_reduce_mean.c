/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_reduce_mean_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_REDUCEMEAN_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_reduce_mean_context_t), sizeof(float));
}

int layer_cpu_float_REDUCEMEAN_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_context_t* context = layer->C->context;
	layer_context_t* input_context = layer->inputs[0]->C->context;
	float* in = (float*)input_context->out[0];
	float* o = (float*)context->out[0];
	int axis = (int)RTE_FETCH_INT32(layer->blobs[0]->blob, 0);

	return r;
}
void layer_cpu_float_REDUCEMEAN_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
