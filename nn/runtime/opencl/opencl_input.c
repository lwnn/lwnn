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
	LAYER_CL_CONTEXT_MEMBER;
	cl_mem in;
} layer_cl_input_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_INPUT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	layer_cl_input_context_t* context;

	context = rte_cl_create_layer_context(nn, layer,
				OPENCL_PATH "input.cl", "input",
				sizeof(layer_cl_input_context_t), &r);

	if(0 == r)
	{
		NNLOG(NN_DEBUG, ("%s dims: [%dx%dx%dx%d] -> [1x%dx%dx4]\n",
							layer->name,
							context->nhwc.N, context->nhwc.H,
							context->nhwc.W, context->nhwc.C,
							RTE_CL_NHWC_H(context->nhwc),
							RTE_CL_NHWC_W(context->nhwc)));

		context->out = rte_cl_create_image2d(nn,
					RTE_CL_NHWC_H(context->nhwc),
					RTE_CL_NHWC_W(context->nhwc));
		context->in = NULL;

		if(NULL == context->out)
		{
			r = NN_E_NO_MEMORY;
			rte_cl_destory_layer_context(nn, context);
		}
	}

	if(0 == r)
	{
		layer->C->context = (layer_context_t*)context;
	}

	return r;
}

int layer_cl_INPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_input_context_t* context = (layer_cl_input_context_t*)layer->C->context;
	float* data;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	data = (float*) nn_get_input_data(nn, layer);

	if(NULL != context->in)
	{
		clReleaseMemObject(context->in);
	}

	context->in = rte_cl_create_buffer(nn, NHWC_SIZE(context->nhwc), data);

	r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHWC, 2,
					sizeof(cl_mem), &(context->in),
					sizeof(cl_mem), &(context->out));

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer);
	}

	return r;
}

void layer_cl_INPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_input_context_t* context = (layer_cl_input_context_t*)layer->C->context;

	if(NULL != context)
	{
		if(NULL != context->in)
		{
			clReleaseMemObject(context->in);
		}
		if(NULL != context->out)
		{
			clReleaseMemObject(context->out);
		}
		rte_cl_destory_layer_context(nn, context);
	}
}
#endif /* DISABLE_RUNTIME_OPENCL */
