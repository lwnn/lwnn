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
} layer_cl_reshape_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_RESHAPE_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_cl_reshape_context_t* context;

	r = rte_cl_create_layer_common(nn, layer,
			OPENCL_PATH "reshape.cl", "reshape", NULL,
			sizeof(layer_cl_reshape_context_t));

	if(0 == r)
	{
		context = (layer_cl_reshape_context_t*)layer->C->context;
		nn_request_scratch(nn, NHWC_SIZE(context->nhwc)*sizeof(float));
	}

	return r;
}
int layer_cl_RESHAPE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_reshape_context_t* context = (layer_cl_reshape_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;

	size_t sz = NHWC_SIZE(context->nhwc);
	float* data;

	input_context = (layer_cl_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	if( (0 == (input_context->nhwc.C&0x3)) &&
		(0 == (context->nhwc.C&0x3)) )
	{
		r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHWC, 6,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(int), &(input_context->nhwc.N),
					sizeof(int), &(input_context->nhwc.H),
					sizeof(int), &(input_context->nhwc.W),
					sizeof(int), &(input_context->nhwc.C));
		if(0 == r)
		{
			r = rte_cl_execute_layer(nn, layer, RTE_GWT_W_H_C, FALSE, NULL);
		}
	}
	else
	{
		data = (float*)nn->scratch.area;
		r = rte_cl_image2d_copy_out(nn, input_context->out[0], data, &(input_context->nhwc));
		if(0 == r)
		{
			r = rte_cl_image2d_copy_in(nn, context->out[0], data, &(context->nhwc));
		}
	}

	return r;
}
void layer_cl_RESHAPE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
