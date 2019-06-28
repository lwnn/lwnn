/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	cl_mem input;
	NHWC_t nhwc;
} layer_opencl_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_opencl_INPUT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	layer_opencl_context_t* context;

	context = malloc(sizeof(layer_opencl_context_t));

	if(context != NULL)
	{
		r = layer_get_NHWC(layer, &context->nhwc);
		if(0 != r)
		{
			free(context);
		}
	}
	else
	{
		r = NN_E_NO_MEMORY;
	}

	if(0 == r)
	{
		int H = context->nhwc.H;
		int W = context->nhwc.W;
		int C = OPENCL_ROUNDUP4(context->nhwc.C);
		W = W*(C>>2);

		NN_LOG(NN_DEBUG,("%s dims: [%dx%dx%dx%d] -> [%dx%dx%dx4]\n", layer->name,
				context->nhwc.N, context->nhwc.H, context->nhwc.W, context->nhwc.C,
				context->nhwc.N,H,W));


		context->input = runtime_opencl_create_image2d(nn, H, W);

		if(NULL == context->input)
		{
			r = NN_E_NO_MEMORY;
		}
	}

	if(0 == r)
	{
		layer->C->context = context;
	}

	return r;
}

int layer_opencl_INPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	return r;
}

int layer_opencl_INPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	return r;
}
#endif /* DISABLE_RUNTIME_OPENCL */
