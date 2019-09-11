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
	LAYER_CPU_Q16_CONTEXT_MEMBER;
} layer_cpu_q16_concat_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q16_CONCAT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q16_concat_context_t* context;
	const layer_t** input = layer->inputs;
	int8_t Q = LAYER_Q(*input);

	input++;
	while(((*input) != NULL) && (0 == r))
	{	/* concat all input layers */

		if(Q != LAYER_Q(*input))
		{
			r = NN_E_INPUTS_Q_MISMATCH;
		}
		input ++;
	}

	if(0 == r)
	{
		r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q16_concat_context_t), sizeof(int16_t));
	}

	return r;
}
int layer_cpu_q16_CONCAT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q16_concat_context_t* context = (layer_cpu_q16_concat_context_t*)layer->C->context;
	int axis = RTE_FETCH_INT32(layer->blobs[1]->blob, 0);

	r = alg_concat(nn, layer, axis, context->out[0], rte_cpu_fetch_out0, sizeof(int16_t));

	return r;
}
void layer_cpu_q16_CONCAT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q16 */
