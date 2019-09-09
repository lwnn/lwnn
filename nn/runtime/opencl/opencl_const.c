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
	cl_mem C;
} layer_cl_const_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_CONST_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_cl_const_context_t* context;

	r = rte_cl_create_layer_context(nn, layer, NULL, NULL, sizeof(layer_cl_const_context_t), 1);

	if(0 == r)
	{
		context = (layer_cl_const_context_t*)layer->C->context;

		context->C = rte_cl_create_image2d_from_blob(nn, layer->blobs[0]);
	}
}

int layer_cl_CONST_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_const_context_t* context = (layer_cl_const_context_t*)layer->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n",layer->name));

	context->out[0] = context->C;

	return r;
}

void layer_cl_CONST_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_const_context_t* context;

	context = (layer_cl_const_context_t*)layer->C->context;

	if(NULL != context)
	{
		rte_cl_destory_memory(context->C);
	}
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
