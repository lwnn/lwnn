/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
	cl_mem W;
} layer_cl_prelu_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_PRELU_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_prelu_context_t* context;

	r = rte_cl_create_layer_common(nn, layer,
				OPENCL_PATH "prelu.cl", "prelu", NULL,
				sizeof(layer_cl_prelu_context_t));
	if(0 == r)
	{
		context = (layer_cl_prelu_context_t*)layer->C->context;

		context->W = rte_cl_create_image2d_from_blob(nn, layer->blobs[0]);
		if(NULL == context->W)
		{
			r = NN_E_NO_MEMORY;
		}

		if(0 != r)
		{
			rte_cl_destory_layer_context(nn, layer);
		}
	}

	return r;
}

int layer_cl_PRELU_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_prelu_context_t* context = (layer_cl_prelu_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context;
	int nC4 = (context->nhwc.C+3)>>2;

	input_context = (layer_cl_context_t*)input->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	r = rte_cl_set_layer_args(nn, layer, 0, 4,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->W),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(int), &nC4);

	if(0 == r)
	{
		r = rte_cl_execute_layer(nn, layer, RTE_GWT_CL_W_H, FALSE, NULL);
	}

	return r;
}

void layer_cl_PRELU_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_prelu_context_t* context;

	context = (layer_cl_prelu_context_t*)layer->C->context;

	if(NULL != context)
	{
		rte_cl_destory_memory(context->W);
	}

	rte_cl_destory_layer_context(nn, layer);
}
#endif /* DISABLE_RUNTIME_OPENCL */
