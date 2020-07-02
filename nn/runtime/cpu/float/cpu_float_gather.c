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
} layer_cpu_float_gather_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_GATHER_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_gather_context_t), sizeof(float));
}

int layer_cpu_float_GATHER_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	layer_cpu_float_gather_context_t* context = (layer_cpu_float_gather_context_t*)layer->C->context;
	layer_cpu_context_t* inputA_context = (layer_cpu_context_t*)layer->inputs[0]->C->context;
	layer_cpu_context_t* inputB_context = (layer_cpu_context_t*)layer->inputs[1]->C->context;
	layer_cpu_context_t* inputC_context = (layer_cpu_context_t*)layer->inputs[2]->C->context;
	float *data = (float*)inputA_context->out[0];
	float *indices = (float*)inputB_context->out[0];
	int axis = ((int32_t*)inputC_context->out[0])[0];
	float *O = (float*)context->out[0];
	size_t n_block = NHWC_SIZE(inputB_context->nhwc);
	size_t stride = 1;
	size_t i;

	for(i = axis+1; i <= 3; i++) {
		stride *= RTE_FETCH_INT32(&(inputA_context->nhwc), i);
	}

	for(i=0; i<n_block; i++) {
		memcpy(O+stride*i, data+stride*((int)indices[i]), stride*sizeof(float));
	}

	return r;
}

void layer_cpu_float_GATHER_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
