/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q16
#include "../runtime_cpu.h"

#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q16_CONTEXT_MEMBER;
#if defined (ARM_MATH_DSP)
	rte_cpu_buffer_t* bufferA;
#endif
} layer_cpu_q16_conv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void convolve_HWC_q15_fast_nonsquare(const q15_t * Im_in,
									const uint16_t dim_im_in_x,
									const uint16_t dim_im_in_y,
									const uint16_t ch_im_in,
									const q15_t * wt,
									const uint16_t ch_im_out,
									const uint16_t dim_kernel_x,
									const uint16_t dim_kernel_y,
									const uint16_t padding_x,
									const uint16_t padding_y,
									const uint16_t stride_x,
									const uint16_t stride_y,
									const q15_t * bias,
									const uint16_t bias_shift,
									const uint16_t out_shift,
									q15_t * Im_out,
									const uint16_t dim_im_out_x,
									const uint16_t dim_im_out_y)
{
	uint16_t  i, j, k, l, m, n;
	int       conv_out;
	int16_t in_row, in_col;

	for (i = 0; i < ch_im_out; i++)
	{
		for (j = 0; j < dim_im_out_y; j++)
		{
			for (k = 0; k < dim_im_out_x; k++)
			{
				conv_out = ((q31_t)bias[i] << bias_shift) + NN_ROUND(out_shift);
				for (m = 0; m < dim_kernel_y; m++)
				{
					for (n = 0; n < dim_kernel_x; n++)
					{
						in_row = stride_y * j + m - padding_y;
						in_col = stride_x * k + n - padding_x;
						if (in_row >= 0 && in_col >= 0 && in_row < dim_im_in_y && in_col < dim_im_in_x)
						{
							for (l = 0; l < ch_im_in; l++)
							{
								conv_out +=
									Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in +
										  l] * wt[i * ch_im_in * dim_kernel_x * dim_kernel_y + (m * dim_kernel_x +
																							n) * ch_im_in + l];
							}
						}
					}
				}
				Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = (q15_t) __SSAT((conv_out >> out_shift), 16);
			}
		}
	}
}
static int convolve(const int16_t * Im_in,  /* input image */
		const int dim_im_in_x,  /* input image dimension x */
		const int dim_im_in_y,  /* input image dimension y */
		const int ch_im_in, /* number of input image channels */
		const int16_t * wt, /* kernel weights */
		const int ch_im_out, /* number of filters, i.e., output image channels */
		const int dim_kernel_x, /* filter kernel size x */
		const int dim_kernel_y, /* filter kernel size y */
		const int padding_x,    /* padding sizes x */
		const int padding_y,    /* padding sizes y */
		const int stride_x, /* stride x */
		const int stride_y, /* stride y */
		const int16_t * bias,   /* bias */
		const int8_t bias_shift,
		const int8_t out_shift,
		int16_t * Im_out, /* output image */
		const int dim_im_out_x, /* output image dimension x */
		const int dim_im_out_y,  /* output image dimension y */
		void* bufferA
		)
{
	int r = 0;
	/* check if can use optimized function
	 *	ch_im_in is multiple of 2
	 * ch_im_out is multiple of 2 */
	if (((ch_im_in&0x01) == 0) &&
			 ((ch_im_out&0x01) == 0) )
	{
		if((dim_im_in_x == dim_im_in_y) &&
				(dim_kernel_x == dim_kernel_y) &&
				(padding_x == padding_y) &&
				(stride_x == stride_y))
		{
			r = arm_convolve_HWC_q15_fast(
					Im_in,
					dim_im_in_x,
					ch_im_in,
					wt,
					ch_im_out,
					dim_kernel_x,
					padding_x,
					stride_x,
					bias,
					bias_shift,
					out_shift,
					Im_out,
					dim_im_out_x,
					bufferA, NULL);
		}
		else
		{
			r = arm_convolve_HWC_q15_fast_nonsquare(
					Im_in,
					dim_im_in_x,
					dim_im_in_y,
					ch_im_in,
					wt,
					ch_im_out,
					dim_kernel_x,
					dim_kernel_y,
					padding_x,
					padding_y,
					stride_x,
					stride_y,
					bias,
					bias_shift,
					out_shift,
					Im_out,
					dim_im_out_x,
					dim_im_out_y,
					bufferA,
					NULL);
		}
	}
	else
	{	/* none optimized */
		if ((dim_im_in_x == dim_im_in_y) &&
			(dim_kernel_x == dim_kernel_y) &&
			(padding_x == padding_y) &&
			(stride_x == stride_y))
		{	/* none opt square shape */
			r = arm_convolve_HWC_q15_basic(
					Im_in,
					dim_im_in_x,
					ch_im_in,
					wt,
					ch_im_out,
					dim_kernel_x,
					padding_x,
					stride_x,
					bias,
					bias_shift,
					out_shift,
					Im_out,
					dim_im_out_x,
					bufferA,
					NULL);
		}
		else
		{	/* none opt none square shape */
			convolve_HWC_q15_fast_nonsquare(
								Im_in,
								dim_im_in_x,
								dim_im_in_y,
								ch_im_in,
								wt,
								ch_im_out,
								dim_kernel_x,
								dim_kernel_y,
								padding_x,
								padding_y,
								stride_x,
								stride_y,
								bias,
								bias_shift,
								out_shift,
								Im_out,
								dim_im_out_x,
								dim_im_out_y);
		}
	}

	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q16_CONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int* ints;
	layer_cpu_q16_conv2d_context_t* context;

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q16_conv2d_context_t), sizeof(int16_t));

#if defined (ARM_MATH_DSP)
	if(0 == r)
	{
		context = (layer_cpu_q16_conv2d_context_t*)layer->C->context;

		ints = (int*)layer->blobs[1]->dims;	/* W in format FHWC */

		context->bufferA = rte_cpu_create_buffer(nn, layer, 2*ints[1]*ints[2]*ints[3]*sizeof(q15_t));

		if(NULL == context->bufferA)
		{
			rte_cpu_destory_layer_context(nn, layer);
		}
		else
		{
			rte_cpu_release_buffer(context->bufferA);
		}
	}
#endif

	return r;
}

int layer_cpu_q16_CONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q16_conv2d_context_t* context = (layer_cpu_q16_conv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q16_context_t* input_context = (layer_cpu_q16_context_t*)input->C->context;
	int16_t *IN = (int16_t*)input_context->out[0];
	int16_t *O = (int16_t*)context->out[0];
	int16_t *weights = (int16_t*)layer->blobs[1]->blob;
	int16_t *bias = (int16_t*)layer->blobs[2]->blob;
	int knlX, knlY, padX, padY, strideX, strideY;
	int8_t wQ, bQ;
	int* ints;

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);

	ints = (int*)layer->blobs[1]->dims;
	knlY = ints[1];
	knlX = ints[2];

	ints = (int*)layer->blobs[3]->blob;
	padY = ints[0];
	padX = ints[1];
	strideY = ints[4];
	strideX = ints[5];

	wQ = (int8_t)ints[6];
	bQ = (int8_t)ints[7];

	NNLOG(NN_DEBUG, (" kernel=[%d %d], pads=[%d %d], strides=[%d %d], %dx%d+%d -> %d\n",
			knlY, knlX, padY, padX, strideY, strideX,
			LAYER_Q(input), wQ, bQ, LAYER_Q(layer)));

	for(batch=0; (batch<input_context->nhwc.N) && (0 == r); batch++)
	{
		r = convolve(IN+batch_sizeIn*batch,
			input_context->nhwc.W,
			input_context->nhwc.H,
			input_context->nhwc.C,
			weights,
			context->nhwc.C,
			knlX, knlY,
			padX, padY,
			strideX, strideY,
			bias,
			wQ+LAYER_Q(input)-bQ,
			wQ+LAYER_Q(input)-LAYER_Q(layer),
			O+batch_sizeO*batch,
			context->nhwc.W,
			context->nhwc.H,
#if defined (ARM_MATH_DSP)
			context->bufferA->data
#else
			NULL
#endif
			);
	}
	return r;
}

void layer_cpu_q16_CONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q16 */
