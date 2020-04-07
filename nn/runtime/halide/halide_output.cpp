/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include "runtime_halide.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_HALIDE_CONTEXT_MEMBER;
} layer_halide_output_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_halide_OUTPUT_init(const nn_t* nn, const layer_t* layer)
{
	return rte_halide_create_layer_context(nn, layer, sizeof(layer_halide_output_context_t), 0);
}

int layer_halide_OUTPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_halide_context_t* context = (layer_halide_context_t*)layer->inputs[0]->C->context;
	Halide::Buffer<float>* in = (Halide::Buffer<float>*)context->out[0];
	float* data;

	data = (float*) nn_get_output_data(nn, layer);
	if(NULL != data)
	{
		std::copy(in->begin(), in->end(), data);
	}
	else
	{
		r = NN_E_NO_OUTPUT_BUFFER_PROVIDED;
	}

	return r;
}

void layer_halide_OUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_halide_output_context_t* context = (layer_halide_output_context_t*)layer->C->context;

	if(NULL != context)
	{
		rte_halide_destory_layer_context(nn, layer);
	}
}
