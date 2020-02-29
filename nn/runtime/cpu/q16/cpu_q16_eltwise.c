/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q16
#include "../runtime_cpu.h"

#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q16_CONTEXT_MEMBER;
} layer_cpu_q16_eltwise_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void layer_cpu_q16_max(int16_t* A, int16_t* B, int16_t* O, size_t sz)
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

static void layer_cpu_q16_min(int16_t* A, int16_t* B, int16_t* O, size_t sz)
{
	size_t i;
	for(i=0; i<sz; i++)
	{
		if(A[i] < B[i])
		{
			O[i] = A[i];
		}
		else
		{
			O[i] = B[i];
		}
	}
}

static void layer_cpu_q16_add(int16_t* A, int16_t* B, int16_t* O, size_t sz, const int8_t out_shift)
{
	size_t i;
	if(0 == out_shift)
	{
		arm_add_q15(A, B, O, sz);
	}
	else
	{
		for(i=0; i<sz; i++)
		{
			O[i] = (q15_t) __SSAT((((int32_t)A[i] + B[i]) >> out_shift), 16);
		}
	}
}

static int layer_cpu_q16_eltwise_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q16_eltwise_context_t), sizeof(int16_t));
}

static int layer_cpu_q16_eltwise_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q16_eltwise_context_t* context = (layer_cpu_q16_eltwise_context_t*)layer->C->context;
	const layer_t* inputA = layer->inputs[0];
	const layer_t* inputB = layer->inputs[1];
	layer_cpu_q16_context_t* inputA_context;
	layer_cpu_q16_context_t* inputB_context;
	size_t sz = NHWC_SIZE(context->nhwc);
	int16_t* A;
	int16_t* B;
	int16_t* O;

	inputA_context = (layer_cpu_q16_context_t*)inputA->C->context;
	inputB_context = (layer_cpu_q16_context_t*)inputB->C->context;

	A = (int16_t*)inputA_context->out[0];
	B = (int16_t*)inputB_context->out[0];
	O = (int16_t*)context->out[0];

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));
	assert(LAYER_Q(inputA) == LAYER_Q(inputB));

	switch(layer->op)
	{
		case L_OP_MAXIMUM:
			layer_cpu_q16_max(A, B, O, sz);
			break;
		case L_OP_ADD:
			layer_cpu_q16_add(A, B, O, sz, LAYER_Q(layer)-LAYER_Q(inputA));
			break;
		case L_OP_MINIMUM:
			layer_cpu_q16_min(A, B, O, sz);
			break;
		default:
			r = NN_E_INVALID_LAYER;
			break;
	}

	return r;
}

static void layer_cpu_q16_eltwise_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q16_MAXIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q16_eltwise_init(nn, layer);
}

int layer_cpu_q16_MAXIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q16_eltwise_execute(nn, layer);
}

void layer_cpu_q16_MAXIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q16_eltwise_deinit(nn, layer);
}

int layer_cpu_q16_MINIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q16_eltwise_init(nn, layer);
}

int layer_cpu_q16_MINIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q16_eltwise_execute(nn, layer);
}

void layer_cpu_q16_MINIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q16_eltwise_deinit(nn, layer);
}


int layer_cpu_q16_ADD_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q16_eltwise_init(nn, layer);
}

int layer_cpu_q16_ADD_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q16_eltwise_execute(nn, layer);
}

void layer_cpu_q16_ADD_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q16_eltwise_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_Q16 */
