/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
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
	LAYER_CPU_DYNMIC_SHAPE_COMMON_MEMBER;
	alg_broadcast_t broadcast;
	layer_context_t* inputA_context;
	layer_context_t* inputB_context;
} layer_cpu_float_eltwise_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
DEF_ALG_ELTWISE(float, MAX)
DEF_ALG_ELTWISE(float, MIN)
DEF_ALG_ELTWISE(float, ADD)
DEF_ALG_ELTWISE(float, SUB)
DEF_ALG_ELTWISE(float, MUL)

DEF_ALG_BROADCAST_ONE(float, MAX)
DEF_ALG_BROADCAST_ONE(float, MIN)
DEF_ALG_BROADCAST_ONE(float, ADD)
DEF_ALG_BROADCAST_ONE(float, SUB)
DEF_ALG_BROADCAST_ONE(float, MUL)

DEF_ALG_BROADCAST_CHANNEL(float, MAX)
DEF_ALG_BROADCAST_CHANNEL(float, MIN)
DEF_ALG_BROADCAST_CHANNEL(float, ADD)
DEF_ALG_BROADCAST_CHANNEL(float, SUB)
DEF_ALG_BROADCAST_CHANNEL(float, MUL)

static int layer_cpu_float_eltwise_init(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_eltwise_context_t* context;
	int r;

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_eltwise_context_t), sizeof(float));

	if(0 == r) {
		layer_cpu_float_eltwise_context_t* context = (layer_cpu_float_eltwise_context_t*)layer->C->context;
		context->broadcast = ALG_BROADCAST_NONE;
		context->inputA_context = (layer_context_t*)layer->inputs[0]->C->context;
		context->inputB_context = (layer_context_t*)layer->inputs[1]->C->context;
		r = alg_broadcast_prepare(&(context->inputA_context), &(context->inputB_context), &(context->broadcast));
	}

	return r;
}

static int layer_cpu_float_eltwise_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_eltwise_context_t* context = (layer_cpu_float_eltwise_context_t*)layer->C->context;
	size_t sz;
	float* A;
	float* B;
	float* O;

#ifndef DISABLE_DYNAMIC_SHAPE
  rte_cpu_dynamic_shape_copy(layer, (layer_cpu_context_t*)context->inputA_context);
  sz = NHWC_SIZE(context->nhwc);
  if(NULL == context->out[0]) {
	r = alg_broadcast_prepare(&(context->inputA_context), &(context->inputB_context), &(context->broadcast));
  }
  if( 0 == r) {
	r = rte_cpu_dynamic_memory(&context->out[0], sz, &context->allocated, sizeof(float));
  }
#endif
  if(0 == r) {
	A = (float*)context->inputA_context->out[0];
	B = (float*)context->inputB_context->out[0];
	O = (float*)context->out[0];

	switch(layer->op+context->broadcast)
	{
		case L_OP_MAXIMUM:
			alg_eltwise_MAX_float(A, B, O, sz);
			break;
		case L_OP_MAXIMUM+ALG_BROADCAST_ONE:
			alg_broadcast_one_MAX_float(A, B[0], O, sz);
			break;
		case L_OP_MAXIMUM+ALG_BROADCAST_CHANNEL:
			alg_broadcast_channel_MAX_float(A, B, O, sz, context->nhwc.C);
			break;
		case L_OP_ADD:
			alg_eltwise_ADD_float(A, B, O, sz);
			break;
		case L_OP_ADD+ALG_BROADCAST_ONE:
			alg_broadcast_one_ADD_float(A, B[0], O, sz);
			break;
		case L_OP_ADD+ALG_BROADCAST_CHANNEL:
			alg_broadcast_channel_ADD_float(A, B, O, sz, context->nhwc.C);
			break;
		case L_OP_MINIMUM:
			alg_eltwise_MIN_float(A, B, O, sz);
			break;
		case L_OP_MINIMUM+ALG_BROADCAST_ONE:
			alg_broadcast_one_MIN_float(A, B[0], O, sz);
			break;
		case L_OP_MINIMUM+ALG_BROADCAST_CHANNEL:
			alg_broadcast_channel_MIN_float(A, B, O, sz, context->nhwc.C);
			break;
		case L_OP_MUL:
			alg_eltwise_MUL_float(A, B, O, sz);
			break;
		case L_OP_MUL+ALG_BROADCAST_ONE:
			alg_broadcast_one_MUL_float(A, B[0], O, sz);
			break;
		case L_OP_MUL+ALG_BROADCAST_CHANNEL:
			alg_broadcast_channel_MUL_float(A, B, O, sz, context->nhwc.C);
			break;
		default:
			r = NN_E_INVALID_LAYER;
			break;
	}
  }
	return r;
}

static void layer_cpu_float_eltwise_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_dynamic_free(layer);
	rte_cpu_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_MAXIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_init(nn, layer);
}

int layer_cpu_float_MAXIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_execute(nn, layer);
}

void layer_cpu_float_MAXIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_eltwise_deinit(nn, layer);
}

int layer_cpu_float_MINIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_init(nn, layer);
}

int layer_cpu_float_MINIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_execute(nn, layer);
}

void layer_cpu_float_MINIMUM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_eltwise_deinit(nn, layer);
}

int layer_cpu_float_ADD_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_init(nn, layer);
}

int layer_cpu_float_ADD_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_execute(nn, layer);
}

void layer_cpu_float_ADD_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_eltwise_deinit(nn, layer);
}

int layer_cpu_float_MUL_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_init(nn, layer);
}

int layer_cpu_float_MUL_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_eltwise_execute(nn, layer);
}

void layer_cpu_float_MUL_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_eltwise_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_FLOAT */
