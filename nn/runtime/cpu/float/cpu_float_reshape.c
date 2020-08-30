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
	LAYER_CPU_DYNMIC_SHAPE_COMMON_MEMBER;
} layer_cpu_float_reshape_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_RESHAPE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_reshape_context_t), sizeof(float));
}

int layer_cpu_float_RESHAPE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_reshape_context_t* context = (layer_cpu_float_reshape_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	size_t sz;
	float* IN = (float*)input_context->out[0];

	rte_cpu_dynamic_reshape(layer, input_context);
	r = rte_cpu_dynamic_memory(&context->out[0], NHWC_SIZE(context->nhwc), &context->allocated, sizeof(float));
	if( 0 == r) {
		sz = NHWC_SIZE(context->nhwc);
		memcpy(context->out[0], IN, sz*sizeof(float));
	}

	return r;
}

void layer_cpu_float_RESHAPE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
