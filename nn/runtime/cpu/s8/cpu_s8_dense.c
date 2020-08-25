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
} layer_cpu_s8_dense_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_s8_DENSE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_s8_dense_context_t), sizeof(int8_t));
}

int layer_cpu_s8_DENSE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_s8_dense_context_t* context = (layer_cpu_s8_dense_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_s8_context_t* input_context = (layer_cpu_s8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	int8_t *weights = (int8_t*)layer->blobs[1]->blob;
	int32_t *bias = (int32_t*)layer->blobs[2]->blob;
	int8_t wQ;
	cmsis_nn_fc_params fc_params;
	cmsis_nn_per_tensor_quant_params quant_params;
	cmsis_nn_dims filter_dims;

	wQ = RTE_FETCH_INT32(layer->blobs[3]->blob, 0);

	fc_params.input_offset = LAYER_Z(input);
	fc_params.output_offset = -LAYER_Z(layer);
	fc_params.filter_offset = RTE_FETCH_INT32(layer->blobs[3]->blob, 1);
	fc_params.activation.min = RTE_FETCH_INT32(layer->blobs[3]->blob, 3);
	fc_params.activation.max = INT8_MAX;

	quant_params.multiplier = RTE_FETCH_INT32(layer->blobs[3]->blob, 2);
	quant_params.shift = -(wQ+LAYER_Q(input)-LAYER_Q(layer));

	filter_dims.n = layer->blobs[1]->dims[1]; /* col_dim */

	NNLOG(NN_DEBUG, (" *[%dx%d] %d -> %d\n",
			layer->blobs[1]->dims[1], layer->blobs[1]->dims[0],
			LAYER_Q(input), LAYER_Q(layer)));

	r = arm_fully_connected_s8(NULL,
				&fc_params,
				&quant_params,
				(cmsis_nn_dims*)&(input_context->nhwc),
				IN,
				(cmsis_nn_dims*)&filter_dims,
				weights,
				(cmsis_nn_dims*)layer->blobs[2]->dims,
				bias,
				(cmsis_nn_dims*)&(context->nhwc),
				O);

	return r;
}

void layer_cpu_s8_DENSE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_S8 */
