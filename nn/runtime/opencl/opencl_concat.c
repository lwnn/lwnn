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
} layer_cl_concat_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
void* cl_concat_fetch_out0(const nn_t* nn, const layer_t* layer)
{
	int r;
	float* pout = (float*)nn->scratch.area;
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;

	r = rte_cl_image2d_copy_out(nn, context->out[0], pout, &(context->nhwc));

	if(0 != r)
	{
		pout = NULL;
	}

	return pout;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_CONCAT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int axis = RTE_FETCH_INT32(layer->blobs[0]->blob, 0);
	const layer_t** input = layer->inputs;
	const char* program = OPENCL_PATH "concat.cl";
	const char* kernel;
	layer_cl_concat_context_t* context;
	layer_cl_context_t* input_context;
	size_t scratch_size = 0;

	switch(axis)
	{
		case 0:
			kernel = "concat_batch";
			break;
		case 1:
			kernel = "concat_height";
			break;
		case 2:
			kernel = "concat_width";
			break;
		case 3:
			kernel = "concat_depth";
			while((0==r) && ((*input) != NULL))
			{
				input_context = (layer_cl_context_t*)(*input)->C->context;
				if(0 != (input_context->nhwc.C&0x03))
				{	/* fall back to CPU */
					program = NULL;
					kernel = NULL;
				}

				if(NHWC_SIZE(input_context->nhwc) > scratch_size)
				{
					scratch_size = NHWC_SIZE(input_context->nhwc);
				}

				input++;
			}
			break;
		default:
			r = NN_E_NOT_SUPPORTED;
			break;
	}

	if(0 == r)
	{
		r = rte_cl_create_layer_common(nn, layer,
				program, kernel,
				sizeof(layer_cl_concat_context_t));
	}

	if(0 == r)
	{
		context = (layer_cl_concat_context_t*)layer->C->context;
		if(0 != scratch_size)
		{
			scratch_size += NHWC_SIZE(context->nhwc);
			nn_request_scratch(nn, scratch_size*sizeof(float));
		}
	}

	return r;
}
int layer_cl_CONCAT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_concat_context_t* context = (layer_cl_concat_context_t*)layer->C->context;
	const layer_t** input = layer->inputs;
	layer_cl_context_t* input_context;

	int axis = RTE_FETCH_INT32(layer->blobs[0]->blob, 0);
	int offset = 0;
	int in_stride;

	NNLOG(NN_DEBUG, ("execute %s: axis=%d\n", layer->name, axis));

	while((0==r) && ((*input) != NULL) && (NULL != context->kernel))
	{
		input_context = (layer_cl_context_t*)(*input)->C->context;
		in_stride = RTE_FETCH_INT32(&(input_context->nhwc), axis);

		NNLOG(NN_DEBUG, ("cl concat %s, in stride=%d\n", (*input)->name, in_stride));
		r = rte_cl_set_layer_args(nn, layer, RTE_CL_ARGS_WITH_NHWC, 4,
					sizeof(cl_mem), &(input_context->out[0]),
					sizeof(cl_mem), &(context->out[0]),
					sizeof(int), &offset,
					sizeof(int), &in_stride);
		if(0 == r)
		{
			r = rte_cl_execute_layer(nn, layer, RTE_GWT_CL_W_H, TRUE, &(input_context->nhwc));
		}

		offset += in_stride;
		input ++;
	}

	if(NULL == context->kernel)
	{	/* fall back to CPU algorithm */
		void* pout = (void*)(((size_t)nn->scratch.area) + nn->scratch.size - NHWC_SIZE(context->nhwc)*sizeof(float));

		r = alg_concat(nn, layer, axis, pout, cl_concat_fetch_out0, sizeof(float));

		if(0 == r)
		{
			r = rte_cl_image2d_copy_in(nn, context->out[0], pout, &(context->nhwc));
		}
	}



	return r;
}
void layer_cl_CONCAT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
