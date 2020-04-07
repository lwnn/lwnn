/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include "runtime_halide.h"
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_HALIDE_CONTEXT_MEMBER;
} layer_halide_input_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_halide_INPUT_init(const nn_t* nn, const layer_t* layer)
{
	return rte_halide_create_layer_common(nn, layer, sizeof(layer_halide_input_context_t));
}

int layer_halide_INPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_halide_input_context_t* context = (layer_halide_input_context_t*)layer->C->context;
	float* data;
	Halide::Buffer<float>* in = (Halide::Buffer<float>*)context->out[0];

	data = (float*) nn_get_input_data(nn, layer);
	if(NULL != data)
	{
		std::copy(data, data+NHWC_SIZE(context->nhwc), in->begin());
	}
	else
	{
		r = NN_E_NO_INPUT_BUFFER_PROVIDED;
	}

	return r;
}

void layer_halide_INPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_halide_destory_layer_context(nn, layer);
}
