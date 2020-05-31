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
	float* scale = (float*)layer->blobs[0]->blob;
	float* bias = (float*)layer->blobs[1]->blob;
	float* var = (float*)layer->blobs[2]->blob;
	float* mean = (float*)layer->blobs[3]->blob;
	float *IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];
	float epsilon = RTE_FETCH_FLOAT(layer->blobs[4]->blob, 0);
	int nC = context->nhwc.N*context->nhwc.H*context->nhwc.W;
	int i,c;
	float X;
	int is_training = FALSE;

	/* s * (x - mean) / np.sqrt(var + epsilon) + bias */
	/* if is_training is True, mean/var is dynamic calculated
	 * a = x.reshape(-1,32)
	 * mean = np.sum(a, axis=0)/a.shape[0]
	 * var = np.sum(np.power(a-mean, 2), axis=0)/a.shape[0] */
	if(layer->inputs[1] != NULL) {
		if((int)RTE_FETCH_FLOAT(layer->inputs[1]->C->context->out[0], 0) != 0) {
			is_training = TRUE;
		}
	}

	if(is_training) {
		for(c=0; c<context->nhwc.C; c++) {
			float mean_ = 0;
			float var_ = 0;
			for(i=0; i<nC; i++) {
				X = IN[i*context->nhwc.C+c];
				mean_ += X;
			}
			mean_ = mean_/nC;

			for(i=0; i<nC; i++) {
				X = IN[i*context->nhwc.C+c];
				var_ += (X-mean_)*(X-mean_);
			}
			var_ = var_/nC;

			for(i=0; i<nC; i++) {
				X = IN[i*context->nhwc.C+c];
				O[i*context->nhwc.C+c] = scale[c]*(X-mean_)/(sqrt(var_+epsilon)) + bias[c];
			}
		}
	} else {
		for(i=0; i<nC; i++) {
			for(c=0; c<context->nhwc.C; c++) {
				X = IN[i*context->nhwc.C+c];
				O[i*context->nhwc.C+c] = scale[c]*(X-mean[c])/(sqrt(var[c]+epsilon)) + bias[c];
			}
		}
	}
	return r;
}
void layer_cpu_float_BATCHNORM_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
