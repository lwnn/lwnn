/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_deconv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void deconvolve_HWC_ref_nonsquare(const float * Im_in,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const float * wt,
		const int ch_im_out,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		const float * bias,
		float * Im_out,
		const int dim_im_out_x,
		const int dim_im_out_y)
{
	int i, j, k, l, m, n;
	float conv_out;
	int in_row, in_col;
	int kernel_start_x,kernel_end_x;
	int kernel_start_y,kernel_end_y;
	int in_row_start, in_col_start;

	for (i = 0; i < ch_im_out; i++)
	{
		for (j = 0; j < dim_im_out_y; j++)
		{
			in_row_start = (j/stride_y) - (padding_y-(j%stride_y))/stride_y;
			if(in_row_start < 0) {
				in_row_start = 0;
			}
			kernel_start_y = (j%stride_y) - padding_y + (j/stride_y)*stride_y;
			if(kernel_start_y < 0) {
				kernel_start_y = -kernel_start_y;
			} else {
				kernel_start_y = kernel_start_y%stride_y;
			}
			kernel_end_y = dim_kernel_y - (j%stride_y) + (dim_im_in_y-j/stride_y)*stride_y;
			if(kernel_end_y > dim_kernel_y) {
				kernel_end_y = dim_kernel_y - ((kernel_end_y-dim_kernel_y)%stride_y);
			}
			if(kernel_start_y == kernel_end_y) {
				kernel_end_y ++;
			}

			for (k = 0; k < dim_im_out_x; k++)
			{
				conv_out = bias[i];

				in_col_start = (k/stride_x) - (padding_x-(k%stride_x))/stride_x;
				if(in_col_start < 0) {
					in_col_start = 0;
				}
				kernel_start_x = (k%stride_x) - padding_x  + (k/stride_x)*stride_x;
				if(kernel_start_x < 0) {
					kernel_start_x = -kernel_start_x;
				} else {
					kernel_start_x = kernel_start_x%stride_x;
				}
				kernel_end_x = dim_kernel_x - (k%stride_x) + (dim_im_in_x-k/stride_x)*stride_x;
				if(kernel_end_x >= dim_kernel_x) {
					kernel_end_x = dim_kernel_x - ((kernel_end_x-dim_kernel_x)%stride_x);
				}
				if(kernel_start_x == kernel_end_x) {
					kernel_end_x ++;
				}

				for (m = kernel_start_y, in_row = in_row_start; m < kernel_end_y; m+=stride_y, in_row++) {
					for (n = kernel_start_x, in_col = in_col_start; n < kernel_end_x; n+=stride_x, in_col++) {
						if ((in_row >= 0) && (in_col >= 0) &&
							(in_row < dim_im_in_y) && (in_col < dim_im_in_x)) {
							for (l = 0; l < ch_im_in; l++) {
								conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + l] *
									wt[i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in + l];
							}
						}
					}
				}

				Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = conv_out;
			}
		}
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_DECONV2D_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_deconv2d_context_t), sizeof(float));
}
int layer_cpu_float_DECONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_deconv2d_context_t* context = (layer_cpu_float_deconv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	float *IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];
	float *weights = (float*)layer->blobs[0]->blob;
	float *bias = (float*)layer->blobs[1]->blob;
	int knlX, knlY, padX, padY, strideX, strideY;
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

	/* TODO: here is experience */
	if(input_context->nhwc.H == context->nhwc.H) {
		padY = 1;
	} else {
		padY = knlY - 1;
	}

	if(input_context->nhwc.W == context->nhwc.W) {
		padX = 1;
	} else {
		padX = knlX - 1;
	}

	NNLOG(NN_DEBUG, ("execute %s: kernel=[%d %d], pads=[%d %d], strides=[%d %d]\n",
			layer->name,
			knlY, knlX, padY, padX, strideY, strideX));

	for(batch=0; batch<input_context->nhwc.N; batch++)
	{
		deconvolve_HWC_ref_nonsquare(IN+batch_sizeIn*batch,
			input_context->nhwc.W,
			input_context->nhwc.H,
			input_context->nhwc.C,
			weights,
			context->nhwc.C,
			knlX, knlY,
			padX, padY,
			strideX, strideY,
			bias,
			O+batch_sizeO*batch,
			context->nhwc.W,
			context->nhwc.H);
	}

	return r;
}
void layer_cpu_float_DECONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
