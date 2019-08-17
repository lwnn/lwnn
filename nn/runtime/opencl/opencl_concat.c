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
} layer_cl_concat_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_CONCAT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int axis = RTE_FETCH_INT32(layer->blobs[0]->blob, 0);
	const layer_t** input = layer->inputs;
	const char* kernel;
	layer_cl_context_t* input_context;

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
				{
					NNLOG(NN_ERROR, ("%s's input %s: depth=%d is not 4 aligned\n",
							layer->name, (*input)->name, input_context->nhwc.C));
					r = NN_E_CL_DEPTH_NOT_4_ALIGNED;
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
				OPENCL_PATH "concat.cl", kernel,
				sizeof(layer_cl_concat_context_t));
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

	while((0==r) && ((*input) != NULL))
	{
		input_context = (layer_cl_context_t*)(*input)->C->context;
		in_stride = RTE_FETCH_INT32(&(input_context->nhwc), axis);

		NNLOG(NN_DEBUG, ("concat %s, in stride=%d\n", (*input)->name, in_stride));
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

	return r;
}
void layer_cl_CONCAT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
