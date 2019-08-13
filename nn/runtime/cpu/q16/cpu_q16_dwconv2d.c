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
} layer_cpu_q16_dwconv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void depthwise_convolve_HWC_q15_fast_nonsquare(const q15_t * Im_in,
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
	int  conv_out;
	int  i_out_y, i_out_x, i_ch_out;
	int  i_ker_y, i_ker_x;
	int in_row, in_col;

	for (i_out_y = 0; i_out_y < dim_im_out_y; i_out_y++)
	{
		for (i_out_x = 0; i_out_x < dim_im_out_x; i_out_x++)
		{
			for (i_ch_out = 0; i_ch_out < ch_im_out; i_ch_out++)
			{
				conv_out = ((q31_t)(bias[i_ch_out]) << bias_shift) + NN_ROUND(out_shift);
				for (i_ker_y = 0; i_ker_y < dim_kernel_y; i_ker_y++)
				{
					for (i_ker_x = 0; i_ker_x < dim_kernel_x; i_ker_x++)
					{
						in_row = stride_y * i_out_y + i_ker_y - padding_y;
						in_col = stride_x * i_out_x + i_ker_x - padding_x;
						if ((in_row >= 0) && (in_col >= 0) && (in_row < dim_im_in_y) && (in_col < dim_im_in_x))
						{
							conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + i_ch_out] *
								wt[dim_kernel_y*dim_kernel_x*i_ch_out + i_ker_y*dim_kernel_x + i_ker_x];
						}
					}
				}
				Im_out[(i_out_y * dim_im_out_x + i_out_x) * ch_im_out + i_ch_out] =
					(q15_t) __SSAT((conv_out >> out_shift), 16);
			}
		}
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q16_DWCONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int* ints;
	layer_cpu_q16_dwconv2d_context_t* context;

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q16_dwconv2d_context_t), sizeof(int16_t));

	if(0 == r)
	{
		context = (layer_cpu_q16_dwconv2d_context_t*)layer->C->context;

		ints = (int*)layer->blobs[2]->blob;
		context->Q = (int8_t)ints[8];
	}

	return r;
}

int layer_cpu_q16_DWCONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q16_dwconv2d_context_t* context = (layer_cpu_q16_dwconv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q16_context_t* input_context = (layer_cpu_q16_context_t*)input->C->context;
	int16_t *IN = (int16_t*)input_context->out[0];
	int16_t *O = (int16_t*)context->out[0];
	int16_t *weights = (int16_t*)layer->blobs[0]->blob;
	int16_t *bias = (int16_t*)layer->blobs[1]->blob;
	int knlX, knlY, padX, padY, strideX, strideY;
	int8_t wQ, bQ;
	int* ints;

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);

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
		depthwise_convolve_HWC_q15_fast_nonsquare(IN+batch_sizeIn*batch,
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
			O+batch_sizeO*batch,
			context->nhwc.W,
			context->nhwc.H
			);
	}
	return r;
}

void layer_cpu_q16_DWCONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q16 */
