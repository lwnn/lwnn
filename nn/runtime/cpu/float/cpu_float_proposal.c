/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
	float* anchors;
	size_t n_anchors;
} layer_cpu_float_proposal_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
int __weak rpn_generate_anchors(const nn_t* nn, const layer_t* layer, float** anchors, size_t* n_anchors)
{
	return NN_E_NOT_IMPLEMENTED;
}
int __weak rpn_proposal_forward(const nn_t* nn, const layer_t* layer, float* anchors, size_t n_anchors)
{
	return NN_E_NOT_IMPLEMENTED;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_PROPOSAL_init(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_proposal_context_t * context;

	int r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_proposal_context_t), sizeof(float));

	if(0 == r) {
		context = (layer_cpu_float_proposal_context_t*) layer->C->context;
		r = rpn_generate_anchors(nn, layer, &context->anchors, &context->n_anchors);
	}

	return r;
}

int layer_cpu_float_PROPOSAL_execute(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_proposal_context_t* context = (layer_cpu_float_proposal_context_t*) layer->C->context;
	NNLOG(NN_DEBUG, ("execute %s\n", layer->name));
	return rpn_proposal_forward(nn, layer, context->anchors, context->n_anchors);
}
void layer_cpu_float_PROPOSAL_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_proposal_context_t * context = (layer_cpu_float_proposal_context_t*) layer->C->context;

	if(context != NULL) {
		if(context->anchors != NULL) {
			free(context->anchors);
		}
	}
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
