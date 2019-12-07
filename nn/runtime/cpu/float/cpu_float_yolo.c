/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
#include "yolo.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_yolo_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_YOLO_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	#ifndef DISABLE_RTE_FALLBACK
	layer_context_t* input_context = (layer_context_t*)layer->inputs[0]->C->context;
	size_t scratch_size;
	#endif

	r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_yolo_context_t), sizeof(float));
	#ifndef DISABLE_RTE_FALLBACK
	if(0 == r)
	{
		if((nn->runtime_type != RUNTIME_CPU) ||
			(nn->network->type != NETWORK_TYPE_FLOAT))
		{
			scratch_size = NHWC_SIZE(input_context->nhwc)*sizeof(float) + sizeof(float*);
			nn_request_scratch(nn, scratch_size);
		}
	}
	#endif

	return r;
}

int layer_cpu_float_YOLO_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_yolo_context_t* context = (layer_cpu_float_yolo_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;
	int num = layer->blobs[0]->dims[0];
	int classes = RTE_FETCH_FLOAT(layer->blobs[2]->blob, 0);

	NNLOG(NN_DEBUG, ("execute %s\n",layer->name));

	r = yolo_forward(context->out[0], input_context->out[0], &input_context->nhwc, num, classes);

	return r;
}

void layer_cpu_float_YOLO_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
