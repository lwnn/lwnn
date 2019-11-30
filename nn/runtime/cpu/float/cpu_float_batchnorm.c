/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
#include <math.h>
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_batchnorm_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_BATCHNORM_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_batchnorm_context_t), sizeof(float));
}
int layer_cpu_float_BATCHNORM_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_context_t* context = (layer_cpu_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	float* scale = layer->blobs[0]->blob;
	float* bias = layer->blobs[1]->blob;
	float* var = layer->blobs[2]->blob;
	float* mean = layer->blobs[3]->blob;
	float *IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];
	float epsilon = RTE_FETCH_FLOAT(layer->blobs[4]->blob, 0);
	int nC = context->nhwc.N*context->nhwc.H*context->nhwc.W;
	int i,c;
	float X;

	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));

	/* s * (x - mean) / np.sqrt(var + epsilon) + bias */
	for(i=0; i<nC; i++)
	{
		for(c=0; c<context->nhwc.C; c++)
		{
			X = IN[i*context->nhwc.C+c];
			O[i*context->nhwc.C+c] = scale[c]*(X-mean[c])/(sqrt(var[c]+epsilon)) + bias[c];
		}
	}

	return r;
}
void layer_cpu_float_BATCHNORM_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
