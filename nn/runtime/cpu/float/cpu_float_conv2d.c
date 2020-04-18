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
#ifndef DISABLE_DYNAMIC_SHAPE
	size_t max;
#endif
} layer_cpu_float_conv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
void __weak convolve_HWC_ref_nonsquare(const float * Im_in,  /* input image */
		const int dim_im_in_x,  /* input image dimension x */
		const int dim_im_in_y,  /* input image dimension y */
		const int ch_im_in, /* number of input image channels */
		const float * wt, /* kernel weights */
		const int ch_im_out, /* number of filters, i.e., output image channels */
		const int dim_kernel_x, /* filter kernel size x */
		const int dim_kernel_y, /* filter kernel size y */
		const int padding_x,    /* padding sizes x */
		const int padding_y,    /* padding sizes y */
		const int stride_x, /* stride x */
		const int stride_y, /* stride y */
		const float * bias,   /* bias */
		float * Im_out, /* output image */
		const int dim_im_out_x, /* output image dimension x */
		const int dim_im_out_y,  /* output image dimension y */
		layer_activation_type_t act
		)
{
	int i, j, k, l, m, n;
	float conv_out;
	int in_row, in_col;

	for (i = 0; i < ch_im_out; i++)
	{
		for (j = 0; j < dim_im_out_y; j++)
		{
			for (k = 0; k < dim_im_out_x; k++)
			{
				conv_out = bias[i];
				for (m = 0; m < dim_kernel_y; m++) {
					for (n = 0; n < dim_kernel_x; n++) {
						in_row = stride_y * j + m - padding_y;
						in_col = stride_x * k + n - padding_x;
						if ((in_row >= 0) && (in_col >= 0) &&
							(in_row < dim_im_in_y) && (in_col < dim_im_in_x))
						{
							for (l = 0; l < ch_im_in; l++) {
								conv_out += Im_in[(in_row * dim_im_in_x + in_col) * ch_im_in + l] *
									wt[i * ch_im_in * dim_kernel_y * dim_kernel_x + (m * dim_kernel_x + n) * ch_im_in + l];
							}
						}
					}
				}
				switch(act) {
					case L_ACT_RELU:
						if(conv_out<0) conv_out = 0;
						break;
					case L_ACT_LEAKY:
						if(conv_out<0) conv_out = 0.1*conv_out;
						break;
					default:
						break;
				}
				Im_out[i + (j * dim_im_out_x + k) * ch_im_out] = conv_out;
			}
		}
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_CONV2D_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_conv2d_context_t), sizeof(float));
}
int layer_cpu_float_CONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_conv2d_context_t* context = (layer_cpu_float_conv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	float *IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];
	float *weights = (float*)layer->blobs[0]->blob;
	float *bias = (float*)layer->blobs[1]->blob;
	int knlX, knlY, padX, padY, strideX, strideY;
	int* ints;
	layer_activation_type_t act;

	size_t batch;
	size_t batch_sizeIn;
	size_t batch_sizeO;

	ints = (int*)layer->blobs[0]->dims;
	knlY = ints[1];
	knlX = ints[2];

	ints = (int*)layer->blobs[2]->blob;
	padY = ints[0];
	padX = ints[1];
	strideY = ints[4];
	strideX = ints[5];
	act = ints[6];

	r = rte_cpu_dynamic_conv2d(layer, (layer_cpu_context_t*)context, input_context,
				&padY, &padX, strideY, strideX, knlY, knlX,
				(void**)&O, &context->max, sizeof(float));
	if(0 == r) {

	batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);
	NNLOG(NN_DEBUG, (" kernel=[%d %d], pads=[%d %d], strides=[%d %d]\n",
			knlY, knlX, padY, padX, strideY, strideX));

	for(batch=0; batch<input_context->nhwc.N; batch++)
	{
		convolve_HWC_ref_nonsquare(IN+batch_sizeIn*batch,
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
			context->nhwc.H,
			act);
	}
	}
	return r;
}
void layer_cpu_float_CONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_dynamic_free(layer);
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
