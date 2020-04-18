/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
	cl_mem scale;
	cl_mem bias;
	cl_mem mean;
	cl_mem var;
} layer_cl_batchnorm_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_BATCHNORM_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_batchnorm_context_t* context;

	r = rte_cl_create_layer_common(nn, layer,
			OPENCL_PATH "batchnorm.cl", "batchnorm", NULL,
			sizeof(layer_cl_batchnorm_context_t));

	if(0 == r)
	{
		context = (layer_cl_batchnorm_context_t*)layer->C->context;
		context->scale = rte_cl_create_image2d_from_blob(nn, layer->blobs[0]);
		context->bias = rte_cl_create_image2d_from_blob(nn, layer->blobs[1]);
		context->var = rte_cl_create_image2d_from_blob(nn, layer->blobs[2]);
		context->mean = rte_cl_create_image2d_from_blob(nn, layer->blobs[3]);

		if( (NULL == context->scale) ||
			(NULL == context->bias) ||
			(NULL == context->mean) ||
			(NULL == context->var) )
		{
			r = NN_E_NO_MEMORY;
		}
	}

	return r;
}

int layer_cl_BATCHNORM_set_args(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_batchnorm_context_t* context = (layer_cl_batchnorm_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context = (layer_cl_context_t*)input->C->context;
	float epsilon = RTE_FETCH_FLOAT(layer->blobs[4]->blob, 0);
	int nC4 = (context->nhwc.C+3)>>2;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	r = rte_cl_set_layer_args(nn, layer, 0, 8,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->scale),
					sizeof(cl_mem), &(context->bias),
					sizeof(cl_mem), &(context->mean),
					sizeof(cl_mem), &(context->var),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(float), &epsilon,
					sizeof(int), &nC4
					);

	return r;
}

int layer_cl_BATCHNORM_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_execute_layer(nn, layer, RTE_GWT_CL_W_H, FALSE, NULL);
}

void layer_cl_BATCHNORM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_batchnorm_context_t* context = (layer_cl_batchnorm_context_t*)layer->C->context;

	if(context != NULL)
	{
		rte_cl_destory_memory(context->scale);
		rte_cl_destory_memory(context->bias);
		rte_cl_destory_memory(context->mean);
		rte_cl_destory_memory(context->var);
	}
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
