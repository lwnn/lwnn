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
} layer_cl_output_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_OUTPUT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	layer_cl_output_context_t* context;

	context = rte_cl_create_layer_context(nn, layer,
				OPENCL_PATH "output.cl", "output",
				sizeof(layer_cl_output_context_t), &r);

	if(0 == r)
	{
		NNLOG(NN_DEBUG, ("%s dims: [%dx%dx%dx%d]\n",
							layer->name,
							context->nhwc.N, context->nhwc.H,
							context->nhwc.W, context->nhwc.C));

		context->out = rte_cl_create_buffer(nn,
							RTE_NHWC_SIZE(context->nhwc),
							NULL);

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

int layer_cl_OUTPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	cl_int errNum;
	layer_cl_output_context_t* context = (layer_cl_output_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;
	float* data;

	input_context = (layer_cl_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	data = (float*) nn_get_input_data(nn, layer);

	r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHWC, 2,
					sizeof(cl_mem), &(input_context->out),
					sizeof(cl_mem), &(context->out));

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer);
	}
	return r;
}

void layer_cl_OUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_output_context_t* context = (layer_cl_output_context_t*)layer->C->context;

	if(NULL != context)
	{
		if(NULL != context->out)
		{
			clReleaseMemObject(context->out);
		}
		rte_cl_destory_layer_context(nn, context);
	}
}

#endif /* DISABLE_RUNTIME_OPENCL */
