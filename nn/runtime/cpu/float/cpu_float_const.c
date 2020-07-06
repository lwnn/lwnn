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
} layer_cpu_float_const_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_CONST_init(const nn_t* nn, const layer_t* layer)
{
	int r = rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_float_const_context_t), 1);

	if(0 == r) {
		layer->C->context->dtype = layer->blobs[0]->dtype;
	}

	return r;
}

int layer_cpu_float_CONST_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_const_context_t* context = (layer_cpu_float_const_context_t*)layer->C->context;

	context->out[0] = layer->blobs[0]->blob;

	return r;
}

void layer_cpu_float_CONST_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
