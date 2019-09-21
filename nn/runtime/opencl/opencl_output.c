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

	r = rte_cl_create_layer_context(nn, layer,
				OPENCL_PATH "output.cl", "output", NULL,
				sizeof(layer_cl_output_context_t), 1);

	if(0 == r)
	{
		context = (layer_cl_output_context_t*)layer->C->context;

		RTE_CL_LOG_LAYER_SHAPE(layer);

		context->out[0] = rte_cl_create_buffer(nn,
							NHWC_SIZE(context->nhwc),
							NULL);

		if(NULL == context->out[0])
		{
			r = NN_E_NO_MEMORY;
			rte_cl_destory_layer_context(nn, layer);
		}
	}

	return r;
}

int layer_cl_OUTPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_output_context_t* context = (layer_cl_output_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;
	float* data;

	input_context = (layer_cl_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	data = (float*) nn_get_output_data(nn, layer);

	if(NULL != data)
	{
		r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHWC, 2,
						sizeof(cl_mem), &(input_context->out[0]),
						sizeof(cl_mem), &(context->out[0]));
	}
	else
	{
		r = NN_E_NO_OUTPUT_BUFFER_PROVIDED;
	}

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer, RTE_GWT_W_H_C, FALSE, NULL);
	}

	if(0 == r)
	{
		r = rte_cl_read_buffer(nn, context->out[0], data, NHWC_SIZE(context->nhwc));
	}

	return r;
}

void layer_cl_OUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
