/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
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
} layer_cpu_float_layernorm_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_LAYER_NORM_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_layernorm_context_t), sizeof(float));
}

int layer_cpu_float_LAYER_NORM_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_context_t* context = (layer_cpu_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	const float* gamma = (float*)layer->blobs[0]->blob;
	const float* beta = (float*)layer->blobs[1]->blob;
	float *IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];
	float epsilon = RTE_FETCH_FLOAT(layer->blobs[2]->blob, 0);;
	int nC = context->nhwc.N*context->nhwc.H*context->nhwc.W;
	int C = context->nhwc.C;
	int i,c;
	float X;
	float mean;
	float var;

	for(i=0; i<nC; i++) {
		mean = 0;
		for(c=0; c<C; c++) {
			X = IN[i*C+c];
			mean += X;
		}
		mean = mean/C;

		var = 0;
		for(c=0; c<C; c++) {
			X = IN[i*C+c];
			var += (X-mean)*(X-mean);
		}
		var = sqrt(var/C + epsilon);

		for(c=0; c<C; c++) {
			X = IN[i*C+c];
			O[i*C+c] = gamma[c]*(X-mean)/var + beta[c];
		}
	}

	return r;
}

void layer_cpu_float_LAYER_NORM_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
