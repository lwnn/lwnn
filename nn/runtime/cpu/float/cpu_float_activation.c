/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
	LAYER_CPU_DYNMIC_SHAPE_COMMON_MEMBER;
} layer_cpu_float_actvation_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void relu_ref(float * out, float * in, size_t size)
{
	size_t  i;

	for (i = 0; i < size; i++)
	{
		if (in[i] < 0) {
			out[i] = 0;
		} else {
			out[i] = in[i];
		}
	}
}

static void prelu_ref(float * out, float * in, size_t size, float* slope, int C)
{
	size_t  i;

	for (i = 0; i < size; i++)
	{
		if (in[i] < 0) {
			out[i] = slope[i%C]*in[i];
		} else {
			out[i] = in[i];
		}
	}
}

static void clip_ref(float* out, float * in, size_t size, float min, float max)
{
	size_t  i;

	for (i = 0; i < size; i++)
	{
		if (in[i] < min) {
			out[i] = min;
		}
		else if (in[i] > max) {
			out[i] = max;
		} else {
			out[i] = in[i];
		}
	}
}

static int layer_cpu_float_activation_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_actvation_context_t), sizeof(float));
}

static int layer_cpu_float_activation_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_actvation_context_t* context = (layer_cpu_float_actvation_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	size_t sz = NHWC_SIZE(input_context->nhwc);
	float* IN;
	float* OUT;

	rte_cpu_dynamic_shape_copy(layer, input_context);
	r = rte_cpu_dynamic_memory(&context->out[0], sz, &context->allocated, sizeof(float));

  if(0 == r) {
	IN = (float*)input_context->out[0];

	OUT = context->out[0];

	switch(layer->op)
	{
		case L_OP_RELU:
			relu_ref(OUT, IN, sz);
			break;
		case L_OP_PRELU:
		{
			float* slope = (float*)layer->blobs[0]->blob;
			assert(layer->blobs[0]->dims[0] == context->nhwc.C);
			prelu_ref(OUT, IN, sz, slope, context->nhwc.C);
			break;
		}
		case L_OP_CLIP:
		{
			float min = RTE_FETCH_FLOAT(layer->blobs[0]->blob, 0);
			float max = RTE_FETCH_FLOAT(layer->blobs[0]->blob, 1);
			clip_ref(OUT, IN, sz, min, max);
			break;
		}
		default:
			r = NN_E_INVALID_LAYER;
			break;
	}
  }
	return r;
}

static void layer_cpu_float_activation_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_dynamic_free(layer);
	rte_cpu_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_RELU_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_activation_init(nn, layer);
}

int layer_cpu_float_RELU_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_activation_execute(nn, layer);
}

void layer_cpu_float_RELU_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_activation_deinit(nn, layer);
}

int layer_cpu_float_PRELU_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_activation_init(nn, layer);
}

int layer_cpu_float_PRELU_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_activation_execute(nn, layer);
}

void layer_cpu_float_PRELU_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_activation_deinit(nn, layer);
}

int layer_cpu_float_CLIP_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_activation_init(nn, layer);
}

int layer_cpu_float_CLIP_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_activation_execute(nn, layer);
}

void layer_cpu_float_CLIP_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_activation_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_FLOAT */
