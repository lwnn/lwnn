/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
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
} layer_cpu_q8_transpose_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_TRANSPOSE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_transpose_context_t), sizeof(int8_t));

}

int layer_cpu_q8_TRANSPOSE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_transpose_context_t* context = (layer_cpu_q8_transpose_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];

	int perm = (int)RTE_FETCH_INT32(layer->blobs[1]->blob, 0);

	NNLOG(NN_DEBUG, ("execute %s: perm=0x%X\n", layer->name, (uint32_t)perm));

	r = alg_transpose(O, IN, &input_context->nhwc, sizeof(int8_t), (alg_transpose_t)perm);

	return r;
}

void layer_cpu_q8_TRANSPOSE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
#ifndef DISABLE_RUNTIME_CPU_S8
int layer_cpu_s8_TRANSPOSE_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_TRANSPOSE_init(nn, layer);
}
int layer_cpu_s8_TRANSPOSE_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_TRANSPOSE_execute(nn, layer);
}
void layer_cpu_s8_TRANSPOSE_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_TRANSPOSE_deinit(nn, layer);
}
#endif
#endif /* DISABLE_RUNTIME_CPU_Q8 */
