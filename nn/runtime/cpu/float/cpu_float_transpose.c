/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
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
} layer_cpu_float_transpose_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_TRANSPOSE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_transpose_context_t), sizeof(float));

}

int layer_cpu_float_TRANSPOSE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_transpose_context_t* context = (layer_cpu_float_transpose_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	float *IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];

	int perm = (int)RTE_FETCH_INT32(layer->blobs[0]->blob, 0);

	NNLOG(NN_DEBUG, (" perm=0x%X\n", (uint32_t)perm));

	r = alg_transpose(O, IN, &input_context->nhwc, sizeof(float), (alg_transpose_t)perm);

	return r;
}

void layer_cpu_float_TRANSPOSE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
