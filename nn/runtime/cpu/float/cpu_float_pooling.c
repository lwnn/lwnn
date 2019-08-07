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
} layer_cpu_float_pool_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void maxpooling(const float * Im_in,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const int ch_im_out,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		float * Im_out,
		const int dim_im_out_x,
		const int dim_im_out_y)
{
	int   i_ch_in, i_x, i_y;
	int   k_x, k_y;

	for (i_ch_in = 0; i_ch_in < ch_im_in; i_ch_in++)
	{
		for (i_y = 0; i_y < dim_im_out_y; i_y++)
		{
			for (i_x = 0; i_x < dim_im_out_x; i_x++)
			{
				float       max = -FLT_MAX;
				for (k_y = i_y * stride_y - padding_y; k_y < i_y * stride_y - padding_y + dim_kernel_y; k_y++)
				{
					for (k_x = i_x * stride_x - padding_x; k_x < i_x * stride_x - padding_x + dim_kernel_x; k_x++)
					{
						if ((k_y >= 0) && (k_x >= 0) && (k_y < dim_im_in_y) && (k_x < dim_im_in_x))
						{
							if (Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)] > max)
							{
								max = Im_in[i_ch_in + ch_im_in * (k_x + k_y * dim_im_in_x)];
							}
						}
					}
				}
				Im_out[i_ch_in + ch_im_in * (i_x + i_y * dim_im_out_x)] = max;
			}
		}
	}
}
static int pooling(const float * Im_in,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const int ch_im_out,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		float * Im_out,
		const int dim_im_out_x,
		const int dim_im_out_y,
		layer_operation_t op)
{
	int r = 0;

	switch(op)
	{
		case L_OP_MAXPOOL:
			maxpooling(Im_in,
				dim_im_in_x,
				dim_im_in_y,
				ch_im_in,
				ch_im_out,
				dim_kernel_x,
				dim_kernel_y,
				padding_x,
				padding_y,
				stride_x,
				stride_y,
				Im_out,
				dim_im_out_x,
				dim_im_out_y);
			break;
		default:
			r = NN_E_INVALID_LAYER;
			break;
	}

	return r;
}
static int layer_cpu_float_pool_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_pool_context_t), sizeof(float));
}

static int layer_cpu_float_pool_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_pool_context_t* context = (layer_cpu_float_pool_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;;
	float* IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];

	int* ints;
	int knlX, knlY, padX, padY, strideX, strideY;
	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);

	ints = (int*)layer->blobs[0]->blob;
	knlY = ints[0];
	knlX = ints[1];
	padY = ints[2];
	padX = ints[3];
	strideY = ints[4];
	strideX = ints[5];

	NNLOG(NN_DEBUG, ("execute %s: kernel=[%d %d], pads=[%d %d], strides=[%d %d]\n", layer->name,
					knlY, knlX, padY, padX, strideY, strideX));


	for(batch=0; batch<input_context->nhwc.N; batch++)
	{
		r = pooling(IN+batch_sizeIn*batch,
				input_context->nhwc.W,
				input_context->nhwc.H,
				input_context->nhwc.C,
				context->nhwc.C,
				knlX, knlY,
				padX, padY,
				strideX, strideY,
				O+batch_sizeO*batch,
				context->nhwc.W,
				context->nhwc.H,
				layer->op
				);
	}
	return r;
}

static void layer_cpu_float_pool_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_MAXPOOL_init(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_pool_init(nn, layer);
}

int layer_cpu_float_MAXPOOL_execute(const nn_t* nn, const layer_t* layer)
{
	return layer_cpu_float_pool_execute(nn, layer);
}

void layer_cpu_float_MAXPOOL_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_pool_deinit(nn, layer);
}
#endif /* DISABLE_RUNTIME_CPU_FLOAT */
