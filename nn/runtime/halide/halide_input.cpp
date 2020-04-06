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
	Halide::Buffer<float>* in;
} layer_halide_input_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_halide_INPUT_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_halide_input_context_t* context;

	r = rte_halide_create_layer_common(nn, layer, sizeof(layer_halide_input_context_t));
	if(0 == r) {
		context = (layer_halide_input_context_t*) layer->C->context;
		context->in = new Halide::Buffer<float>(context->nhwc.C, context->nhwc.W, context->nhwc.H);
		if(NULL != context->in) {
			context->in->allocate();
		} else {
			r = NN_E_NO_MEMORY;
		}
	}

	if(0 == r) {
		Halide::Buffer<float>& in = *(context->in);
		in.set_name(std::string(layer->name)+"_IN");
		Halide::Var y, x, c;
		Halide::Func& func = *(Halide::Func*) context->out[0];
		func(c, x, y) = in(c, x, y);
	}

	return r;
}

int layer_halide_INPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_halide_input_context_t* context = (layer_halide_input_context_t*)layer->C->context;
	float* data;

	data = (float*) nn_get_input_data(nn, layer);
	if(NULL != data)
	{
		std::copy(data, data+NHWC_SIZE(context->nhwc), context->in->begin());
	}
	else
	{
		r = NN_E_NO_INPUT_BUFFER_PROVIDED;
	}

	return r;
}

void layer_halide_INPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_halide_input_context_t* context = (layer_halide_input_context_t*)layer->C->context;

	if(NULL != context)
	{
		if(NULL != context->in)
		{
			delete (Halide::Buffer<float>*)context->in;
		}
		rte_halide_destory_layer_context(nn, layer);
	}
}
