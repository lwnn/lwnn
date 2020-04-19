/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
#include <math.h>
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
	LAYER_CPU_DYNMIC_SHAPE_COMMON_MEMBER;
} layer_cpu_float_softmax_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void softmax_ref(const float * vec_in, const size_t dim_vec, float * p_out)
{
	float     sum;
	float     base;
	size_t   i;
	base = -FLT_MAX;

	/* We first search for the maximum */
	for (i = 0; i < dim_vec; i++)
	{
		if (vec_in[i] > base)
		{
			base = vec_in[i];
		}
	}

	sum = 0;

	for (i = 0; i < dim_vec; i++)
	{
		sum += exp(vec_in[i] - base);
	}

	for (i = 0; i < dim_vec; i++)
	{
		p_out[i] = exp(vec_in[i] - base)/sum;
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_SOFTMAX_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_cpu_float_softmax_context_t* context;
	void* out = nn_get_output_data(nn, layer);

	if(NULL != out)
	{
		r = rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_float_softmax_context_t), 1);
		if(0 == r)
		{
			context = (layer_cpu_float_softmax_context_t*)layer->C->context;
			context->out[0] = out;
		}
	}
	else
	{
		r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_softmax_context_t), sizeof(float));
	}

	return r;
}

int layer_cpu_float_SOFTMAX_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_softmax_context_t* context = (layer_cpu_float_softmax_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	float *IN = (float*)input_context->out[0];
	float *O;
	size_t n_block = input_context->nhwc.N*input_context->nhwc.H*input_context->nhwc.W;
	size_t stride = input_context->nhwc.C;
	size_t i;

  rte_cpu_dynamic_shape_copy(layer, input_context);
  r = rte_cpu_dynamic_memory(&context->out[0], NHWC_SIZE(context->nhwc), &context->allocated, sizeof(float));
  if(0 == r) {
	O = (float*)context->out[0];

	for(i=0; i<n_block; i++)
	{
		softmax_ref(IN+stride*i,
					stride,
					O+stride*i);
	}
  }
	return r;
}

void layer_cpu_float_SOFTMAX_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_dynamic_free(layer);
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
