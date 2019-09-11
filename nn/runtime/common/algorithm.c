/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int alg_concat(const nn_t* nn, const layer_t* layer, int axis,
		void* pout, void* (*fetch_input)(const nn_t* nn, const layer_t* layer),
		size_t type_size)
{
	int r = 0;
	void* pin;
	layer_context_t* context = (layer_context_t*)layer->C->context;
	const layer_t** input = layer->inputs;
	layer_context_t* input_context;

	size_t n_block;
	size_t in_stride;
	size_t out_stride;
	size_t i,j;

	n_block = 1;
	for (i = 0; i < axis; i++)
	{	/* Calculate the number of block to concat. (the other shapes before the concat axis) */
		n_block *= RTE_FETCH_INT32(&(context->nhwc), i);
	}
	out_stride = 1;
	for(j = axis; j <= 3; j++)
	{
		out_stride *= RTE_FETCH_INT32(&(context->nhwc), j);
	}

	NNLOG(NN_DEBUG, ("execute %s[%d %d %d %d]: axis=%d, n_block=%d, out stride=%d\n",
			layer->name,
			context->nhwc.N, context->nhwc.H, context->nhwc.W, context->nhwc.C,
			axis, n_block, out_stride));

	while((*input) != NULL)
	{	/* concat all input layers */
		input_context = (layer_context_t*)(*input)->C->context;
		pin = fetch_input(nn, *input);

		if(NULL == pin)
		{
			r = NN_E_NO_MEMORY;
			break;
		}

		in_stride = 1;
		for(j = axis; j <= 3; j++)
		{
			in_stride *= RTE_FETCH_INT32(&(input_context->nhwc), j);
		}

		NNLOG(NN_DEBUG, ("concat %s[%d %d %d %d], in stride=%d\n",
				(*input)->name,
				input_context->nhwc.N, input_context->nhwc.H,
				input_context->nhwc.W, input_context->nhwc.C,
				in_stride));

		for(i=0; i<n_block; i++)
		{
			memcpy((void*)(((size_t)pout)+i*out_stride*type_size), pin, in_stride*type_size);
			pin = (void*)(((size_t)pin) + in_stride*type_size);
		}
		pout = (void*)(((size_t)pout) + in_stride*type_size);
		input++;
	}

	return r;
}
