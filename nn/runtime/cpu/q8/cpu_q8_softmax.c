/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#if !defined(DISABLE_RUNTIME_CPU_Q8) || !defined(DISABLE_RUNTIME_CPU_S8)
#include "../runtime_cpu.h"

#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
	void* p_out;
} layer_cpu_q8_softmax_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_SOFTMAX_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_cpu_q8_softmax_context_t* context;
	void* out = nn_get_output_data(nn, layer);

	if(NULL != out)
	{
		r = rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_q8_softmax_context_t), 1);
	}
	else
	{
		r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_softmax_context_t), sizeof(int8_t));
	}

	if(0 == r)
	{
		context = (layer_cpu_q8_softmax_context_t*)layer->C->context;
		context->p_out = out;
	}

	return r;
}

int layer_cpu_q8_SOFTMAX_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_softmax_context_t* context = (layer_cpu_q8_softmax_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q8_context_t* input_context = (layer_cpu_q8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	size_t n_block = context->nhwc.N*context->nhwc.H*context->nhwc.W;
	size_t stride = context->nhwc.C;
	size_t i;

	if(NULL == O)
	{
		O = (int8_t*)context->p_out;
		context->out[0] = context->p_out;
	}

	NNLOG(NN_DEBUG, ("execute %s\n",layer->name));

	for(i=0; i<n_block; i++)
	{
		arm_softmax_q7(IN+stride*i,
					stride,
					O+stride*i);
	}

	return r;
}

void layer_cpu_q8_SOFTMAX_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#ifndef DISABLE_RUNTIME_CPU_S8
int layer_cpu_s8_SOFTMAX_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_SOFTMAX_init(nn, layer);
}
int layer_cpu_s8_SOFTMAX_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_SOFTMAX_execute(nn, layer);
}
void layer_cpu_s8_SOFTMAX_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_SOFTMAX_deinit(nn, layer);
}
#endif
#endif /* DISABLE_RUNTIME_CPU_Q8 */
