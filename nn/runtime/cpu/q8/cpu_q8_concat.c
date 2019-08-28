/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#if !defined(DISABLE_RUNTIME_CPU_Q8) || !defined(DISABLE_RUNTIME_CPU_S8)
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
} layer_cpu_q8_concat_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_CONCAT_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	const layer_t** input = layer->inputs;
	int8_t Q = LAYER_Q(*input);

	input++;
	while(((*input) != NULL) && (0 == r))
	{	/* concat all input layers */
		if(Q != LAYER_Q(*input))
		{
			r = NN_E_INPUTS_Q_MISMATCH;
		}
		input ++;
	}

	if(0 == r)
	{
		r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_concat_context_t), sizeof(int8_t));
	}

	return r;
}
int layer_cpu_q8_CONCAT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_concat_context_t* context = (layer_cpu_q8_concat_context_t*)layer->C->context;
	int axis = RTE_FETCH_INT32(layer->blobs[1]->blob, 0);
	const layer_t** input = layer->inputs;
	layer_cpu_q8_context_t* input_context;

	int8_t* pin;
	int8_t* pout = (int8_t*)context->out[0];
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

	NNLOG(NN_DEBUG, ("execute %s: axis=%d, n_block=%d, out stride=%d\n", layer->name, axis, n_block, out_stride));

	while((*input) != NULL)
	{	/* concat all input layers */
		input_context = (layer_cpu_q8_context_t*)(*input)->C->context;
		pin = (int8_t*)input_context->out[0];

		in_stride = 1;
		for(j = axis; j <= 3; j++)
		{
			in_stride *= RTE_FETCH_INT32(&(input_context->nhwc), j);
		}

		NNLOG(NN_DEBUG, ("concat %s, in stride=%d\n", (*input)->name, in_stride));
		for(i=0; i<n_block; i++)
		{
			memcpy(pout+i*out_stride, pin, in_stride);
			pin += in_stride;
		}
		pout += in_stride;
		input++;
	}

	return r;
}
void layer_cpu_q8_CONCAT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#ifndef DISABLE_RUNTIME_CPU_S8
int layer_cpu_s8_CONCAT_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_CONCAT_init(nn, layer);
}
int layer_cpu_s8_CONCAT_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_q8_CONCAT_execute(nn, layer);
}
void layer_cpu_s8_CONCAT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
#endif
#endif /* DISABLE_RUNTIME_CPU_Q8 */
