/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q16
#include "../runtime_cpu.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q16_CONTEXT_MEMBER;
} layer_cpu_q16_batchnorm_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q16_BATCHNORM_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q16_batchnorm_context_t), sizeof(int16_t));
}
int layer_cpu_q16_BATCHNORM_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_context_t* context = (layer_cpu_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	int16_t* scale = (int16_t*)layer->blobs[1]->blob;
	int16_t* bias = (int16_t*)layer->blobs[2]->blob;
	int16_t *IN = (int16_t*)input_context->out[0];
	int16_t *O = (int16_t*)context->out[0];
	int nC = context->nhwc.N*context->nhwc.H*context->nhwc.W;
	int Qs = RTE_FETCH_INT32(layer->blobs[3]->blob, 0);
	int Qb = RTE_FETCH_INT32(layer->blobs[3]->blob, 1);
	int i,c;
	int32_t out;

	for(i=0; i<nC; i++)
	{
		for(c=0; c<context->nhwc.C; c++)
		{
			out = IN[i*context->nhwc.C+c];
			out = ((((int32_t)scale[c]*out)<<Qb) + ((int32_t)bias[c]<<Qs))>>(Qs+Qb);
			O[i*context->nhwc.C+c] = (uint16_t)__SSAT(out, 16);
		}
	}

	return r;
}
void layer_cpu_q16_BATCHNORM_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q16 */
