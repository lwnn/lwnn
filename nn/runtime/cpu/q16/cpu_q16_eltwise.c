/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q16
#include "../runtime_cpu.h"
#include "algorithm.h"
#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q16_CONTEXT_MEMBER;
	alg_broadcast_t broadcast;
	layer_context_t* inputA_context;
	layer_context_t* inputB_context;
} layer_cpu_q16_eltwise_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
DEF_ALG_ELTWISE(int16_t, MAX)
DEF_ALG_ELTWISE(int16_t, MIN)

DEF_ALG_BROADCAST_ONE(int16_t, MAX)
DEF_ALG_BROADCAST_ONE(int16_t, MIN)

DEF_ALG_BROADCAST_CHANNEL(int16_t, MAX)
DEF_ALG_BROADCAST_CHANNEL(int16_t, MIN)

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
	int r =0;
	layer_cpu_q16_eltwise_context_t* context;

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q16_eltwise_context_t), sizeof(int16_t));

	if(0 == r)
	{
		context = (layer_cpu_q16_eltwise_context_t*)layer->C->context;
		context->broadcast = ALG_BROADCAST_NONE;
		context->inputA_context = (layer_context_t*)layer->inputs[0]->C->context;
		context->inputB_context = (layer_context_t*)layer->inputs[1]->C->context;
		r = alg_broadcast_prepare(&(context->inputA_context), &(context->inputB_context), &(context->broadcast));
	}

	return r;
}

static int layer_cpu_q16_eltwise_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q16_eltwise_context_t* context = (layer_cpu_q16_eltwise_context_t*)layer->C->context;
	size_t sz = NHWC_SIZE(context->nhwc);
	int16_t* A;
	int16_t* B;
	int16_t* O;

	A = (int16_t*)context->inputA_context->out[0];
	B = (int16_t*)context->inputB_context->out[0];
	O = (int16_t*)context->out[0];

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));
	assert(LAYER_Q(layer->inputs[0]) == LAYER_Q(layer->inputs[1]));

	switch(layer->op)
	{
		case L_OP_MAXIMUM:
			alg_eltwise_MAX_int16_t(A, B, O, sz);
			break;
		case L_OP_MAXIMUM+ALG_BROADCAST_ONE:
			alg_broadcast_one_MAX_int16_t(A, B[0], O, sz);
			break;
		case L_OP_MAXIMUM+ALG_BROADCAST_CHANNEL:
			alg_broadcast_channel_MAX_int16_t(A, B, O, sz, context->nhwc.C);
			break;
		case L_OP_ADD:
			layer_cpu_q16_add(A, B, O, sz, LAYER_Q(layer)-LAYER_Q(layer->inputs[0]));
			break;
		case L_OP_MINIMUM:
			alg_eltwise_MIN_int16_t(A, B, O, sz);
			break;
		case L_OP_MINIMUM+ALG_BROADCAST_ONE:
			alg_broadcast_one_MIN_int16_t(A, B[0], O, sz);
			break;
		case L_OP_MINIMUM+ALG_BROADCAST_CHANNEL:
			alg_broadcast_channel_MIN_int16_t(A, B, O, sz, context->nhwc.C);
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
