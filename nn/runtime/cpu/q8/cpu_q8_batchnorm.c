/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#if !defined(DISABLE_RUNTIME_CPU_Q8) || !(DISABLE_RUNTIME_CPU_S8)
#include "../runtime_cpu.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
} layer_cpu_q8_batchnorm_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_BATCHNORM_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_batchnorm_context_t), sizeof(int8_t));
}
int layer_cpu_q8_BATCHNORM_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_context_t* context = (layer_cpu_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	int8_t* scale = (int8_t*)layer->blobs[1]->blob;
	int8_t* bias = (int8_t*)layer->blobs[2]->blob;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	int nC = context->nhwc.N*context->nhwc.H*context->nhwc.W;
	int Qs = RTE_FETCH_INT32(layer->blobs[3]->blob, 0);
	int Qb = RTE_FETCH_INT32(layer->blobs[3]->blob, 1);
	int i,c;
	int32_t out;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	for(i=0; i<nC; i++)
	{
		for(c=0; c<context->nhwc.C; c++)
		{
			out = IN[i*context->nhwc.C+c];
			out = ((((int32_t)scale[c]*out)<<Qb) + ((int32_t)bias[c]<<Qs))>>(Qs+Qb);
			O[i*context->nhwc.C+c] = (uint8_t)__SSAT(out, 8);
		}
	}

	return r;
}
void layer_cpu_q8_BATCHNORM_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#ifndef DISABLE_RUNTIME_CPU_S8
int layer_cpu_s8_BATCHNORM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_BATCHNORM_init(nn, layer);
}
int layer_cpu_s8_BATCHNORM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_BATCHNORM_execute(nn, layer);
}
void layer_cpu_s8_BATCHNORM_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
#endif
#endif /* DISABLE_RUNTIME_CPU_Q8 */
