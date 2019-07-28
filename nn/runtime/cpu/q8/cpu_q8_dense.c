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
} layer_cpu_q8_dense_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_DENSE_init(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_dense_context_t* context;

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_dense_context_t), sizeof(int8_t));

	if(0 == r)
	{
		context = (layer_cpu_q8_dense_context_t*)layer->C->context;

		context->Q = RTE_FETCH_INT8(layer->blobs[2]->blob, 2);

#if defined (ARM_MATH_DSP)
		context->bufferA = rte_cpu_create_buffer(nn, layer, RTE_FETCH_INT32(layer->blobs[0]->dims, 0));

		if(NULL == context->bufferA)
		{
			rte_cpu_destory_layer_context(nn, layer);
		}
#endif
	}

	return r;
}

int layer_cpu_q8_DENSE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_dense_context_t* context = (layer_cpu_q8_dense_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q8_context_t* input_context = (layer_cpu_q8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	int8_t *weights = (int8_t*)layer->blobs[0]->blob;
	int8_t *bias = (int8_t*)layer->blobs[1]->blob;
	int8_t wQ, bQ;

	uint16_t dim_vec = (uint16_t)RTE_FETCH_INT32(layer->blobs[0]->dims, 0);
	uint16_t num_of_rows = (uint16_t)RTE_FETCH_INT32(layer->blobs[0]->dims, 1);

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);


	wQ = RTE_FETCH_INT8(layer->blobs[2]->blob, 0);
	bQ = RTE_FETCH_INT8(layer->blobs[2]->blob, 1);

	NNLOG(NN_DEBUG, ("execute %s: [%d %d] %dx%d+%d -> %d\n",
			layer->name,
			dim_vec, num_of_rows,
			input_context->Q, wQ, bQ, context->Q));

	for(batch=0; (batch<input_context->nhwc.N) && (0 == r); batch++)
	{
		r = arm_fully_connected_q7_opt(IN+batch_sizeIn*batch,
				weights,
				dim_vec,
				num_of_rows,
				wQ+input_context->Q-bQ,
				wQ+input_context->Q-context->Q,
				bias,
				O+batch_sizeO*batch,
#if defined (ARM_MATH_DSP)
				context->bufferA->data
#else
				NULL
#endif
				);
	}
	return r;
}

void layer_cpu_q8_DENSE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q8 */
