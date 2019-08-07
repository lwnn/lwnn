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
} layer_cpu_float_dense_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void fully_connected_ref(const float * pV,
						const float * pM,
						const int dim_vec,
						const int num_of_rows,
						const float * bias,
						float * pOut)
{
	int i,j;
	float ip_out;
	for (i = 0; i < num_of_rows; i++)
	{
		ip_out = bias[i];
		for (j = 0; j < dim_vec; j++)
		{
			ip_out += pV[j] * pM[i * dim_vec + j];
		}
		pOut[i] = ip_out;
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_DENSE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_dense_context_t), sizeof(float));

}

int layer_cpu_float_DENSE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_dense_context_t* context = (layer_cpu_float_dense_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	float *IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];
	float *weights = (float*)layer->blobs[0]->blob;
	float *bias = (float*)layer->blobs[1]->blob;

	int num_of_rows = (int)RTE_FETCH_INT32(layer->blobs[0]->dims, 0);
	int dim_vec = (int)RTE_FETCH_INT32(layer->blobs[0]->dims, 1);

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);

	NNLOG(NN_DEBUG, ("execute %s: [%d %d]\n", layer->name, dim_vec, num_of_rows));

	for(batch=0; batch<input_context->nhwc.N; batch++)
	{
		fully_connected_ref(IN+batch_sizeIn*batch,
				weights,
				dim_vec,
				num_of_rows,
				bias,
				O+batch_sizeO*batch);
	}

	return r;
}

void layer_cpu_float_DENSE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
