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
#if defined (ARM_MATH_DSP)
	rte_cpu_buffer_t* bufferA;
#endif
} layer_cpu_q8_dwconv2d_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_DWCONV2D_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
#if defined (ARM_MATH_DSP)
	int* ints;
	layer_cpu_q8_dwconv2d_context_t* context;
#endif

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_dwconv2d_context_t), sizeof(int8_t));

#if defined (ARM_MATH_DSP)
	if(0 == r)
	{
		context = (layer_cpu_q8_dwconv2d_context_t*)layer->C->context;

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

int layer_cpu_q8_DWCONV2D_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_dwconv2d_context_t* context = (layer_cpu_q8_dwconv2d_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q8_context_t* input_context = (layer_cpu_q8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	int8_t *weights = (int8_t*)layer->blobs[1]->blob;
	int8_t *bias = (int8_t*)layer->blobs[2]->blob;
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
		r = arm_depthwise_separable_conv_HWC_q7_nonsquare(
				IN+batch_sizeIn*batch,
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
				context->bufferA->data,
#else
				NULL,
#endif
				NULL
				);
	}
	return r;
}

void layer_cpu_q8_DWCONV2D_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q8 */
