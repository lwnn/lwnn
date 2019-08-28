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

static void layer_cpu_q8_add(int8_t* A, int8_t* B, int8_t* O, size_t sz, const int8_t out_shift)
{
	size_t i;
	if(0 == out_shift)
	{
		arm_add_q7(A, B, O, sz);
	}
	else
	{
		for(i=0; i<sz; i++)
		{
			O[i] = (q7_t) __SSAT(((A[i] + B[i]) >> out_shift), 8);
		}
	}
}

static int layer_cpu_q8_eltwise_init(const nn_t* nn, const layer_t* layer)
{
	int r =0;
	layer_cpu_q8_eltwise_context_t* context;

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_eltwise_context_t), sizeof(int8_t));

	if(0 == r)
	{
		context = (layer_cpu_q8_eltwise_context_t*)layer->C->context;
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
		case L_OP_ADD:
			assert(LAYER_Q(inputA) == LAYER_Q(inputB));
			layer_cpu_q8_add(A, B, O, sz, LAYER_Q(layer)-LAYER_Q(inputA));
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

int layer_cpu_q8_ADD_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_eltwise_init(nn, layer);
}

int layer_cpu_q8_ADD_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_eltwise_execute(nn, layer);
}

void layer_cpu_q8_ADD_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_eltwise_deinit(nn, layer);
}

#ifndef DISABLE_RUNTIME_CPU_S8
int layer_cpu_s8_MAXIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_eltwise_init(nn, layer);
}
int layer_cpu_s8_MAXIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_eltwise_execute(nn, layer);
}
void layer_cpu_s8_MAXIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_eltwise_deinit(nn, layer);
}
#endif
#endif /* DISABLE_RUNTIME_CPU_Q8 */
