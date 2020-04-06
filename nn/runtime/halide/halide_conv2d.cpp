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
	Halide::Buffer<float>* W;
	Halide::Buffer<float>* B;
} layer_halide_conv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_halide_CONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_halide_conv2d_context_t* context;
	layer_halide_context_t* input_context;

	r = rte_halide_create_layer_common(nn, layer, sizeof(layer_halide_conv2d_context_t));

	if(0 == r) {
		input_context = (layer_halide_context_t*)layer->inputs[0]->C->context;
		context = (layer_halide_conv2d_context_t*) layer->C->context;
		context->W = rte_halide_create_buffer_from_blob(nn, layer->blobs[0]);
		if(NULL != context->W)
		{
			context->B = rte_halide_create_buffer_from_blob(nn, layer->blobs[1]);
			if(NULL == context->B)
			{
				context->W = NULL;
				r = NN_E_NO_MEMORY;
			}
		}
		else
		{
			context->B = NULL;
			r = NN_E_NO_MEMORY;
		}
	}

	if(0 == r) {
		int* ints = (int*)layer->blobs[2]->blob;
		const int CI = input_context->nhwc.C, CO = context->nhwc.C;
		const int knlY = layer->blobs[0]->dims[1];
		const int knlX = layer->blobs[0]->dims[2];
		Halide::Buffer<float>& W = *(context->W);
		Halide::Buffer<float>& B = *(context->B);
		W.set_name(std::string(layer->name)+"_W");
		B.set_name(std::string(layer->name)+"_B");
		const int padY = ints[0];
		const int padX = ints[1];
		const int strideY = ints[4];
		const int strideX = ints[5];
		Halide::Var y, x, c;
		Halide::RDom r(0, knlX, 0, knlY, 0, CI);
		Halide::Func& in = *(Halide::Func*) input_context->out[0];
		Halide::Func in_bounded =
			Halide::BoundaryConditions::constant_exterior(in, 0.f,
					{{Halide::Expr(), Halide::Expr()},
					 {0, input_context->nhwc.W},
					 {0, input_context->nhwc.H}});
		Halide::Func& conv = *(Halide::Func*) context->out[0];
		conv(c, x, y) = B(c);
		Halide::Expr in_row = strideY * y + r.y - padY;
		Halide::Expr in_col = strideX * x + r.x - padX;
		conv(c, x, y) += W(r.z, r.x, r.y, c) * in_bounded(r.z, in_col, in_row);
	}

	return r;
}

int layer_halide_CONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	return 0;
}

void layer_halide_CONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_halide_conv2d_context_t* context = (layer_halide_conv2d_context_t*)layer->C->context;

	if(NULL != context)
	{
		if(context->W) delete context->W;
		if(context->B) delete context->B;
		rte_halide_destory_layer_context(nn, layer);
	}
}
