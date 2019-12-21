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
} layer_cpu_s8_dwconv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_s8_DWCONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
#if defined (ARM_MATH_DSP)
	int* ints;
	layer_cpu_s8_dwconv2d_context_t* context;
#endif

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_s8_dwconv2d_context_t), sizeof(int8_t));

#if defined (ARM_MATH_DSP)
	if(0 == r)
	{
		context = (layer_cpu_s8_dwconv2d_context_t*)layer->C->context;

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
int layer_cpu_s8_DWCONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_s8_dwconv2d_context_t* context = (layer_cpu_s8_dwconv2d_context_t*)layer->C->context;
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

	NNLOG(NN_DEBUG, ("execute %s: kernel=[%d %d], pads=[%d %d], strides=[%d %d], Z=%d, Q %d -> %d\n",
			layer->name,
			knlY, knlX, padY, padX, strideY, strideX,
			LAYER_Z(layer), LAYER_Q(input), LAYER_Q(layer)));

	for(batch=0; (batch<input_context->nhwc.N) && (0 == r); batch++)
	{
		r = arm_depthwise_conv_s8_opt(IN+batch_sizeIn*batch,
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
					context->nhwc.W,
					context->nhwc.H,
					-LAYER_Z(layer),
					LAYER_Z(input),
					INT8_MIN,
					INT8_MAX,
					0,0,
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
int layer_cpu_s8_DWCONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_s8_dwconv2d_context_t* context = (layer_cpu_s8_dwconv2d_context_t*)layer->C->context;
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

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);

	uint16_t output_ch = context->nhwc.C;
	uint16_t output_y = context->nhwc.H;
	uint16_t output_x = context->nhwc.W;
	uint16_t input_ch = input_context->nhwc.C;
	uint16_t input_y = input_context->nhwc.H;
	uint16_t input_x = input_context->nhwc.W;

	int ch_mult = output_ch/input_ch;

	assert((output_ch%input_ch) == 0);

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
		int i_out = 0;
		int8_t *in = IN+batch_sizeIn*batch;
		int8_t *out = O+batch_sizeO*batch;

		for (int i_out_y = 0; i_out_y < output_y; i_out_y++)
		{
			const int16_t base_idx_y = (i_out_y * stride_y) - pad_y;
			for (int i_out_x = 0; i_out_x < output_x; i_out_x++)
			{
				const int16_t base_idx_x = (i_out_x * stride_x) - pad_x;
				for (int i_input_ch = 0; i_input_ch < input_ch; i_input_ch++)
				{
					for (int i_ch_mult = 0; i_ch_mult < ch_mult; i_ch_mult++)
					{
						const int idx_out_ch = i_ch_mult + i_input_ch * ch_mult;
						float acc_0;
						/* Condition for kernel start dimension: (base_idx_<x,y> + ker_<x,y>_start) >= 0 */
						const int ker_y_start = MAX(0, -base_idx_y);
						const int ker_x_start = MAX(0, -base_idx_x);
						/* Condition for kernel end dimension: (base_idx_<x,y> + ker_<x,y>_end) < input_<x,y> */
						const int ker_y_end = MIN(kernel_y, input_y - base_idx_y);
						const int ker_x_end = MIN(kernel_x, input_x - base_idx_x);
						int Wq = Oq - Iq - output_shift[idx_out_ch];
						acc_0 = bias[idx_out_ch]*Is/(1<<(Iq+Wq));

						for (int i_ker_y = ker_y_start; i_ker_y < ker_y_end; i_ker_y++)
						{
							const int32_t idx_y = base_idx_y + i_ker_y;
							for (int i_ker_x = ker_x_start; i_ker_x < ker_x_end; i_ker_x++)
							{
								const int32_t idx_x = base_idx_x + i_ker_x;
								int32_t idx_0 = (idx_y * input_x + idx_x) * input_ch + i_input_ch;
								int32_t ker_idx_0 = (i_ker_y * kernel_x + i_ker_x) * (input_ch * ch_mult) + idx_out_ch;

								acc_0 += Is*(in[idx_0] + input_offset)/(1<<Iq) * kernel[ker_idx_0]/(1<<Wq);
							}
						}

						acc_0 = acc_0*(1<<Oq)/Os - out_offset;
						acc_0 = MAX(acc_0, out_activation_min);
						acc_0 = MIN(acc_0, out_activation_max);

						out[i_out++] = acc_0;
					}
				}
			}
		}
	}
	return r;
}
#endif
void layer_cpu_s8_DWCONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_S8 */
