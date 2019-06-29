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
	layer_cl_context_t C;
	cl_mem input;
} layer_cl_input_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_opencl_INPUT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	layer_cl_input_context_t* context;

	context = runtime_opencl_create_context(nn, layer,
				OPENCL_PATH "input.cl", "input",
				sizeof(layer_cl_input_context_t), &r);

	if(0 == r)
	{
		int H = context->C.nhwc.H;
		int W = context->C.nhwc.W;
		int C = OPENCL_ROUNDUP4(context->C.nhwc.C);
		W = W*(C>>2);

		NNLOG(NN_DEBUG,("%s dims: [%dx%dx%dx%d] -> [%dx%dx%dx4]\n", layer->name,
				context->C.nhwc.N, context->C.nhwc.H, context->C.nhwc.W, context->C.nhwc.C,
				context->C.nhwc.N,H,W));

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

	float* data = (float*) nn_get_input_data(nn, layer);

	NNLOG(NN_DEBUG,("%s input: [ %f %f %f %f ...]\n",
			layer->name, data[0], data[1], data[2], data[3]));

	return r;
}

int layer_opencl_INPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	return r;
}
#endif /* DISABLE_RUNTIME_OPENCL */
