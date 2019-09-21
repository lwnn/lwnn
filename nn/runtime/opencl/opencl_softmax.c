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
	void* p_out;
} layer_cl_softmax_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_SOFTMAX_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_cl_softmax_context_t* context;
	void* out = nn_get_output_data(nn, layer);

	r = rte_cl_create_layer_common(nn, layer,
					OPENCL_PATH "softmax.cl", "softmax", NULL,
					sizeof(layer_cl_softmax_context_t));

	if(0 == r)
	{
		context = (layer_cl_softmax_context_t*)layer->C->context;
		context->p_out = out;
	}

	return r;
}
int layer_cl_SOFTMAX_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_softmax_context_t* context = (layer_cl_softmax_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;

	input_context = (layer_cl_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHC, 2,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->out[0]));

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer, RTE_GWT_W_H, FALSE, NULL);
	}

	if((0 == r) && (NULL != context->p_out))
	{
		r = rte_cl_image2d_copy_out(nn, context->out[0], context->p_out, &context->nhwc);
	}

	return r;
}
void layer_cl_SOFTMAX_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
