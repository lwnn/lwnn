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
#define TEMPLATE_ALG_TRANSPOSE_FROM_NHWC_TO_NCHW(DTYPE)					\
static void alg_transpose_from_nhwc_to_nchw_##DTYPE(					\
		void* output,													\
		const void* input,												\
		const NHWC_t *inhwc)											\
{																		\
	int n,h,w,c;														\
	size_t indexI;														\
	const DTYPE##_t* pin = (const DTYPE##_t*)input;						\
	DTYPE##_t* pout = (DTYPE##_t*)output;								\
																		\
	for(n=0; n<inhwc->N; n++)											\
	{																	\
		for(c=0; c<inhwc->C; c++)										\
		{																\
			for(h=0; h<inhwc->H; h++)									\
			{															\
				for(w=0; w<inhwc->W; w++)								\
				{														\
					indexI = ((n*inhwc->H+h)*inhwc->W+w)*inhwc->C+c;	\
					*pout = pin[indexI];								\
					pout++;												\
				}														\
			}															\
		}																\
	}																	\
}

TEMPLATE_ALG_TRANSPOSE_FROM_NHWC_TO_NCHW(int8)
TEMPLATE_ALG_TRANSPOSE_FROM_NHWC_TO_NCHW(int16)
TEMPLATE_ALG_TRANSPOSE_FROM_NHWC_TO_NCHW(int32)
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

int alg_up_sampling(void* pout, void* pin, NHWC_t *outNHWC, NHWC_t *inNHWC, size_t type_size)
{
	int r = 0;
	int x,y,n,i;
	int strideX, strideY;
	void* p_in;
	void* p_out;

	NNLOG(NN_DEBUG, (" [%d %d %d %d] -> [%d %d %d %d]\n",
			inNHWC->N, inNHWC->H, inNHWC->W, inNHWC->C,
			outNHWC->N, outNHWC->H, outNHWC->W, outNHWC->C));

	assert(inNHWC->N==outNHWC->N);
	assert(inNHWC->C==outNHWC->C);

	strideY = outNHWC->H/inNHWC->H;
	strideX = outNHWC->W/inNHWC->W;

	for(n=0; n<inNHWC->N; n++)
	{
		pout = APABO(pout, n*NHWC_SIZE(*outNHWC));
		pin = APABO(pin, n*NHWC_SIZE(*inNHWC));
		for (y=0; y<inNHWC->H; y++)
		{
			for (x=0; x<inNHWC->W; x++)
			{
				/* copy all the channels together. */
				p_in = APABO(pin, (y*inNHWC->W+x)*inNHWC->C*type_size);
				p_out = APABO(pout, (y*strideY*outNHWC->W+x*strideX)*inNHWC->C*type_size);

				/* copy along x axis */
				for(i=0; i<strideX; i++)
				{
					memcpy(APABO(p_out, i*inNHWC->C*type_size), p_in, inNHWC->C*type_size);
				}

				/* duplicate the copied x data into y axis. */
				for(i=1; i<strideY; i++)
				{
					memcpy(APABO(p_out, i*inNHWC->C*outNHWC->W*type_size), p_out, inNHWC->C*strideX*type_size);
				}
			}
		}
	}

	return r;
}

int alg_transpose(void* output, const void* input, const NHWC_t *inhwc, size_t type_size, alg_transpose_t transpose)
{
	int r = 0;
	void (*transpose_func)(void*, const void*, const NHWC_t*) = NULL;

	switch(type_size)
	{
		case 1:
			transpose_func = alg_transpose_from_nhwc_to_nchw_int8;
			break;
		case 2:
			transpose_func = alg_transpose_from_nhwc_to_nchw_int16;
			break;
		case 4:
			transpose_func = alg_transpose_from_nhwc_to_nchw_int32;
			break;
		default:
			r = NN_E_INVALID_PARAMETER;
			break;
	}

	if(0 == r)
	{
		switch(transpose)
		{
			case ALG_TRANSPOSE_FROM_NHWC_TO_NCHW:
				transpose_func(output, input, inhwc);
				break;
			default:
				r = NN_E_INVALID_PARAMETER;
				break;
		}
	}

	return r;
}

