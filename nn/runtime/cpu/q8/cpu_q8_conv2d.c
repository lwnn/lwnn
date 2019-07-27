/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q8
#include "../runtime_cpu.h"

#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
	rte_cpu_buffer_t* bufferA;
} layer_cpu_q8_conv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int convolve(const int8_t * Im_in,  // input image
		const int dim_im_in_x,  // input image dimension x
		const int dim_im_in_y,  // input image dimension y
		const int ch_im_in, // number of input image channels
		const int8_t * wt, // kernel weights
		const int ch_im_out, // number of filters, i.e., output image channels
		const int dim_kernel_x, // filter kernel size x
		const int dim_kernel_y, // filter kernel size y
		const int padding_x,    // padding sizes x
		const int padding_y,    // padding sizes y
		const int stride_x, // stride x
		const int stride_y, // stride y
		const int8_t * bias,   // bias
		const int8_t bias_shift,
		const int8_t out_shift,
		int8_t * Im_out, // output image
		const int dim_im_out_x, // output image dimension x
		const int dim_im_out_y,  // output image dimension y
		void* bufferA
		)
{
	int r = 0;
	/* RGB: ch_im_in = 3, w = h */
	if ( (ch_im_in == 3) &&
			(dim_im_in_x == dim_im_in_y) &&
			(dim_kernel_x == dim_kernel_y) &&
			(padding_x == padding_y) &&
			(stride_x == stride_y)) {
		r = arm_convolve_HWC_q7_RGB(
			Im_in, dim_im_in_x, ch_im_in,
			wt,
			ch_im_out,
			dim_kernel_x, padding_x, stride_x,
			bias, bias_shift,
			out_shift, Im_out, dim_im_out_x,
			(q15_t *)bufferA, NULL);
	}
	/* check if can use optimized function
	 *	ch_im_in is multiple of 4
	 * ch_im_out is multiple of 2 */
	else if (((ch_im_in&0x03) == 0) &&
			 ((ch_im_out&0x01) == 0) )
	{
		if ((dim_kernel_x == 1) && (dim_kernel_y == 1))
		{
			r = arm_convolve_1x1_HWC_q7_fast_nonsquare(
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
		else if((dim_im_in_x == dim_im_in_y) &&
				(dim_kernel_x == dim_kernel_y) &&
				(padding_x == padding_y) &&
				(stride_x == stride_y))
		{
			r = arm_convolve_HWC_q7_fast(
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
			r = arm_convolve_HWC_q7_fast_nonsquare(
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
			r = arm_convolve_HWC_q7_basic(
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
			r = arm_convolve_HWC_q7_basic_nonsquare(
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

	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_CONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int* ints;
	layer_cpu_q8_conv2d_context_t* context;

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_conv2d_context_t), sizeof(int8_t));

	if(0 == r)
	{
		context = (layer_cpu_q8_conv2d_context_t*)layer->C->context;

		ints = (int*)layer->blobs[2]->blob;
		context->Q = (int8_t)ints[8];

#if defined (ARM_MATH_DSP)
		ints = (int*)layer->blobs[0]->dims;	/* W in format FHWC */

		context->bufferA = rte_cpu_create_buffer(nn, layer, 2*ints[1]*ints[2]*ints[3]);

		if(NULL == context->bufferA)
		{
			rte_cpu_destory_layer_context(nn, layer);
		}
#endif
	}

	return r;
}

int layer_cpu_q8_CONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_conv2d_context_t* context = (layer_cpu_q8_conv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q8_context_t* input_context = (layer_cpu_q8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	int8_t *weights = (int8_t*)layer->blobs[0]->blob;
	int8_t *bias = (int8_t*)layer->blobs[1]->blob;
	int knlX, knlY, padX, padY, strideX, strideY;
	int8_t wQ, bQ;
	int* ints;

	size_t batch;
	size_t batch_size = NHWC_BATCH_SIZE(input_context->nhwc);

	ints = (int*)layer->blobs[0]->dims;
	knlY = ints[1];
	knlX = ints[2];

	ints = (int*)layer->blobs[2]->blob;
	padY = ints[0];
	padX = ints[1];
	strideY = ints[4];
	strideX = ints[5];

	wQ = (int8_t)ints[6];
	bQ = (int8_t)ints[7];

	NNLOG(NN_DEBUG, ("execute %s: kernel=[%d %d], pads=[%d %d], strides=[%d %d], %dx%d+%d -> %d\n",
			layer->name,
			knlY, knlX, padY, padX, strideY, strideX,
			input_context->Q, wQ, bQ, context->Q));

	for(batch=0; batch<input_context->nhwc.N; batch++)
	{
		r = convolve(IN+batch_size*batch,
			input_context->nhwc.W,
			input_context->nhwc.H,
			input_context->nhwc.C,
			weights,
			context->nhwc.C,
			knlX, knlY,
			padX, padY,
			strideX, strideY,
			bias,
			wQ+input_context->Q-bQ,
			wQ+input_context->Q-context->Q,
			O+batch_size*batch,
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

void layer_cpu_q8_CONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q8 */
