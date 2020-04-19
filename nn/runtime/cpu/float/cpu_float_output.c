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
} layer_cpu_float_output_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_OUTPUT_init(const nn_t* nn, const layer_t* layer)
{
	void* out = nn_get_output_data(nn, layer);
	layer_cpu_float_output_context_t* context;
	int r = rte_cpu_create_layer_context(nn, layer,
				sizeof(layer_cpu_float_output_context_t), 1);

	if(0 == r)
	{
		context = (layer_cpu_float_output_context_t*)layer->C->context;
		context->out[0] = out;
	}

	return r;
}

int layer_cpu_float_OUTPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_output_context_t* context = (layer_cpu_float_output_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context;
	float* data;

	input_context = (layer_cpu_context_t*)input->C->context;

  rte_cpu_dynamic_shape_copy(layer, input_context);
  r = rte_cpu_dynamic_memory(&context->out[0], NHWC_SIZE(context->nhwc), &context->allocated, sizeof(float));
  if(0 == r) {
	data = (float*)context->out[0];
	if(NULL != data)
	{
		context->nhwc = input_context->nhwc;
		memcpy(data, input_context->out[0], NHWC_SIZE(context->nhwc)*sizeof(float));
	}
	else
	{
		r = NN_E_NO_OUTPUT_BUFFER_PROVIDED;
	}
  }
	return r;
}

void layer_cpu_float_OUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_dynamic_free(layer);
	rte_cpu_destory_layer_context(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_FLOAT */
