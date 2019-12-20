/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_S8
#include "../runtime_cpu.h"

#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_S8_CONTEXT_MEMBER;
#if defined (ARM_MATH_DSP)
	rte_cpu_buffer_t* bufferA;
#endif
} layer_cpu_s8_conv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_s8_CONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
#if defined (ARM_MATH_DSP)
	int* ints;
	layer_cpu_s8_conv2d_context_t* context;
#endif

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_s8_conv2d_context_t), sizeof(int8_t));

#if defined (ARM_MATH_DSP)
	if(0 == r)
	{
		context = (layer_cpu_s8_conv2d_context_t*)layer->C->context;

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
#if 1
int layer_cpu_s8_CONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_s8_conv2d_context_t* context = (layer_cpu_s8_conv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_s8_context_t* input_context = (layer_cpu_s8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	int8_t *weights = (int8_t*)layer->blobs[1]->blob;
	int32_t *bias = (int32_t*)layer->blobs[2]->blob;
	int knlX, knlY, padX, padY, strideX, strideY;
	int* ints;
	int omin;

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

	omin = ints[6];

	NNLOG(NN_DEBUG, ("execute %s: kernel=[%d %d], pads=[%d %d], strides=[%d %d], Z=%d, %d -> %d\n",
			layer->name,
			knlY, knlX, padY, padX, strideY, strideX,
			LAYER_Z(layer), LAYER_Q(input), LAYER_Q(layer)));

	for(batch=0; (batch<input_context->nhwc.N) && (0 == r); batch++)
	{
		r = arm_convolve_s8(IN+batch_sizeIn*batch,
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
					(const int32_t*)layer->blobs[5]->blob,
					(const int32_t*)layer->blobs[4]->blob,
					-LAYER_Z(layer),
					LAYER_Z(input),
					omin,
					INT8_MAX,
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
#else
int layer_cpu_s8_CONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_s8_conv2d_context_t* context = (layer_cpu_s8_conv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_s8_context_t* input_context = (layer_cpu_s8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	int8_t *kernel = (int8_t*)layer->blobs[1]->blob;
	int32_t *bias = (int32_t*)layer->blobs[2]->blob;
	uint16_t kernel_x, kernel_y, pad_x, pad_y, stride_x, stride_y;
	int* ints;
	int out_activation_min;
	int out_activation_max = INT8_MAX;
	int Iq = LAYER_Q(input);
	int Oq = LAYER_Q(layer);
	float Is = (float)LAYER_S(input)/NN_SCALER;
	float Os = (float)LAYER_S(layer)/NN_SCALER;
	int input_offset = LAYER_Z(input);
	int out_offset = LAYER_Z(layer);

	const int32_t *output_shift = (const int32_t*)layer->blobs[5]->blob;
	const int32_t *output_mult = (const int32_t*)layer->blobs[4]->blob;

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);

	ints = (int*)layer->blobs[1]->dims;
	kernel_y = ints[1];
	kernel_x = ints[2];

	ints = (int*)layer->blobs[3]->blob;
	pad_y = ints[0];
	pad_x = ints[1];
	stride_y = ints[4];
	stride_x = ints[5];

	out_activation_min = ints[6];

	NNLOG(NN_DEBUG, ("execute %s: kernel=[%d %d], pads=[%d %d], strides=[%d %d], Z=%d, %d -> %d\n",
			layer->name,
			kernel_y, kernel_x, pad_y, pad_x, stride_y, stride_x,
			LAYER_Z(layer), LAYER_Q(input), LAYER_Q(layer)));

	for(batch=0; (batch<input_context->nhwc.N) && (0 == r); batch++)
	{
		int8_t *in = IN+batch_sizeIn*batch;
		int8_t *out = O+batch_sizeO*batch;
		uint16_t i, j, k, l, m, n;
		float conv_out;
		int16_t in_row, in_col;
		uint16_t output_ch = context->nhwc.C;
		uint16_t output_y = context->nhwc.H;
		uint16_t output_x = context->nhwc.W;
		uint16_t input_ch = input_context->nhwc.C;
		uint16_t input_y = input_context->nhwc.H;
		uint16_t input_x = input_context->nhwc.W;

		for (i = 0; i < output_ch; i++)
		{
			int Wq = Oq - Iq - output_shift[i];
			float bs = Is/(1<<(Iq+Wq));
			float ibias = bias[i]*bs;

			for (j = 0; j < output_y; j++)
			{
				for (k = 0; k < output_x; k++)
				{
					conv_out = ibias;
					for (m = 0; m < kernel_y; m++)
					{
						for (n = 0; n < kernel_x; n++)
						{
							in_row = stride_y * j + m - pad_y;
							in_col = stride_x * k + n - pad_x;
							if (in_row >= 0 && in_col >= 0 && in_row < input_y && in_col < input_x)
							{
								for (l = 0; l < input_ch; l++)
								{
									conv_out +=
										Is*(in[(in_row * input_x + in_col) * input_ch + l] + input_offset)/(1<<Iq) *
										kernel[i * input_ch * kernel_y * kernel_x +
											   (m * kernel_x + n) * input_ch + l]/(1<<Wq);
								}
							}
						}
					}
					conv_out = conv_out*(1<<Oq)/Os - out_offset;
					conv_out = MAX(conv_out, out_activation_min);
					conv_out = MIN(conv_out, out_activation_max);
					out[i + (j * output_x + k) * output_ch] = (q7_t)conv_out;
				}
			}
		}
	}
	return r;
}
#endif
void layer_cpu_s8_CONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_S8 */
