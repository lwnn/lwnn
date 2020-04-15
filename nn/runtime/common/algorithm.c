
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

#define TEMPLATE_ALG_TRANSPOSE_FROM_NCHW_TO_NHWC(DTYPE)					\
static void alg_transpose_from_nchw_to_nhwc_##DTYPE(					\
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
		for(h=0; h<inhwc->H; h++)										\
		{																\
			for(w=0; w<inhwc->W; w++)									\
			{															\
				for(c=0; c<inhwc->C; c++)								\
				{														\
					indexI = ((n*inhwc->C+c)*inhwc->H+h)*inhwc->W+w;	\
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

TEMPLATE_ALG_TRANSPOSE_FROM_NCHW_TO_NHWC(int8)
TEMPLATE_ALG_TRANSPOSE_FROM_NCHW_TO_NHWC(int16)
TEMPLATE_ALG_TRANSPOSE_FROM_NCHW_TO_NHWC(int32)
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
			axis, (int)n_block, (int)out_stride));

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

		NNLOG(NN_DEBUG, ("  concat %s[%d %d %d %d], in stride=%d\n",
				(*input)->name,
				input_context->nhwc.N, input_context->nhwc.H,
				input_context->nhwc.W, input_context->nhwc.C,
				(int)in_stride));

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

int alg_up_sampling(void* pout, void* pin, NHWC_t *outNHWC, NHWC_t *inNHWC, size_t type_size, uint8_t* pmask)
{
	int r = 0;
	int x,y,c,n,i;
	int strideX, strideY;
	void* p_in;
	void* p_out;
	int offset,dx,dy;

	NNLOG(NN_DEBUG, (" [%d %d %d %d] -> [%d %d %d %d]\n",
			inNHWC->N, inNHWC->H, inNHWC->W, inNHWC->C,
			outNHWC->N, outNHWC->H, outNHWC->W, outNHWC->C));

	assert(inNHWC->N==outNHWC->N);
	assert(inNHWC->C==outNHWC->C);

	strideY = outNHWC->H/inNHWC->H;
	strideX = outNHWC->W/inNHWC->W;

	if(pmask)
	{
		memset(pout, 0, NHWC_SIZE(*outNHWC)*type_size);
	}

	for(n=0; n<inNHWC->N; n++)
	{
		pout = APABO(pout, n*NHWC_SIZE(*outNHWC));
		pin = APABO(pin, n*NHWC_SIZE(*inNHWC));
		for (y=0; y<inNHWC->H; y++)
		{
			for (x=0; x<inNHWC->W; x++)
			{
				if(NULL != pmask)
				{
					for(c=0; c<inNHWC->C; c++){
						offset = pmask[(y*inNHWC->W+x)*inNHWC->C+c];
						dy = offset/strideX;
						dx = offset%strideX;
						p_in = APABO(pin, ((y*inNHWC->W+x)*inNHWC->C+c)*type_size);
						p_out = APABO(pout, (((y*strideY+dy)*outNHWC->W+(x*strideX+dx))*inNHWC->C+c)*type_size);
						memcpy(p_out, p_in, type_size);
					}
				}
				else
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
	}

	return r;
}

int alg_transpose(void* output, const void* input, const NHWC_t *inhwc, size_t type_size, alg_transpose_t transpose)
{
	int r = 0;
	void (*transpose_func)(void*, const void*, const NHWC_t*) = NULL;

	switch(type_size|transpose)
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
		case 0x8001:
			transpose_func = alg_transpose_from_nchw_to_nhwc_int8;
			break;
		case 0x8002:
			transpose_func = alg_transpose_from_nchw_to_nhwc_int16;
			break;
		case 0x8004:
			transpose_func = alg_transpose_from_nchw_to_nhwc_int32;
			break;
		default:
			r = NN_E_INVALID_PARAMETER;
			break;
	}

	if(0 == r)
	{
		transpose_func(output, input, inhwc);
	}

	return r;
}

int alg_deconv2d_calculate_position(
		int pos,
		int stride,
		int padding,
		int dim_kernel,
		int dim_in,
		int* in_start,
		int* kernel_start,
		int* kernel_end)
{
	int is_zero = FALSE;
	int of, adj;
	is_zero = FALSE;
	*in_start = pos/stride;
	of = pos%stride;
	*kernel_start = padding - of;
	if(*kernel_start >= 0) {
		adj = NN_MIN(*in_start, *kernel_start/stride);
		*kernel_start -= adj*stride;
		*in_start -= adj;
	} else {
		adj = -*kernel_start + dim_kernel;
		if(adj<=stride) {
			is_zero = TRUE;
		} else {
			adj = NN_MIN(dim_in-1-*in_start, adj/stride);
			*kernel_start += adj*stride;
			*in_start += adj;
		}
	}
	of = dim_kernel - 1 - *kernel_start;
	adj = NN_MIN(dim_in-1-*in_start, of/stride);
	*kernel_end = *kernel_start + adj*stride;

	return is_zero;
}

int alg_deconv2d_calculate_padding(int dim_kernel, int stride, int dim_in, int dim_out)
{
	int padding;

	/* TODO: here is experience */
	if(dim_in == dim_out) {
		padding = 1;
	} else {
		if(dim_kernel < 5) {
			padding = dim_kernel - 1;
		} else {
			if((dim_kernel - stride) >= 2) {
				padding = dim_kernel - 2;
			} else {
				padding = dim_kernel - 1;
			}
		}
	}
	return padding;
}

int alg_broadcast_prepare(layer_context_t** inputA_context, layer_context_t** inputB_context, alg_broadcast_t *broadcast)
{
	int r = 0;
	layer_context_t* tmpC;
	size_t szA = NHWC_SIZE((*inputA_context)->nhwc);
	size_t szB = NHWC_SIZE((*inputB_context)->nhwc);

	*broadcast = ALG_BROADCAST_NONE;
	if(szA > szB) {
		if(1 == szB) {
			*broadcast = ALG_BROADCAST_ONE;
		} else if((*inputA_context)->nhwc.C == szB) {
			*broadcast = ALG_BROADCAST_CHANNEL;
		} else {
			r = NN_E_INVALID_DIMENSION;
		}
	} else if(szA < szB) {
		if(1 == szA) {
			*broadcast = ALG_BROADCAST_ONE;
		} else if((*inputB_context)->nhwc.C == szA) {
			*broadcast = ALG_BROADCAST_CHANNEL;
		} else {
			r = NN_E_INVALID_DIMENSION;
		}
		tmpC = *inputA_context;
		*inputA_context = *inputB_context;
		*inputB_context = tmpC;
	} else {
		/* pass */
	}

	return r;
}
