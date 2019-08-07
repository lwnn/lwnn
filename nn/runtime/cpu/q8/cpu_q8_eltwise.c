/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q8
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
} layer_cpu_q8_eltwise_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void layer_cpu_q8_max(int8_t* A, int8_t* B, int8_t* O, size_t sz)
{
	size_t i;
	for(i=0; i<sz; i++)
	{
		if(A[i] > B[i])
		{
			O[i] = A[i];
		}
		else
		{
			O[i] = B[i];
		}
	}
}
static int layer_cpu_q8_eltwise_init(const nn_t* nn, const layer_t* layer)
{
	int r =0;
	int8_t* int8s;
	layer_cpu_q8_eltwise_context_t* context;

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_eltwise_context_t), sizeof(int8_t));

	if(0 == r)
	{
		context = (layer_cpu_q8_eltwise_context_t*)layer->C->context;

		int8s = (int8_t*)layer->blobs[0]->blob;
		context->Q = int8s[0];
	}

	return r;
}

static int layer_cpu_q8_eltwise_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_eltwise_context_t* context = (layer_cpu_q8_eltwise_context_t*)layer->C->context;
	const layer_t* inputA = layer->inputs[0];
	const layer_t* inputB = layer->inputs[1];
	layer_cpu_q8_context_t* inputA_context;
	layer_cpu_q8_context_t* inputB_context;
	size_t sz = NHWC_SIZE(context->nhwc);
	int8_t* A;
	int8_t* B;
	int8_t* O;

	inputA_context = (layer_cpu_q8_context_t*)inputA->C->context;
	inputB_context = (layer_cpu_q8_context_t*)inputB->C->context;

	A = (int8_t*)inputA_context->out[0];
	B = (int8_t*)inputB_context->out[0];
	O = (int8_t*)context->out[0];

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	switch(layer->op)
	{
		case L_OP_MAXIMUM:
			layer_cpu_q8_max(A, B, O, sz);
			break;
		default:
			r = NN_E_INVALID_LAYER;
			break;
	}

	return r;
}

static void layer_cpu_q8_eltwise_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_MAXIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_eltwise_init(nn, layer);
}

int layer_cpu_q8_MAXIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_eltwise_execute(nn, layer);
}

void layer_cpu_q8_MAXIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_eltwise_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_Q8 */
