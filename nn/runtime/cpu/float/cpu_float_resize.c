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
} layer_cpu_float_resize_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
int CropAndResize_forward_cpu(float* o, const float* in, const float* boxes, const int* indices,
		NHWC_t* onhwc, NHWC_t* inhwc);
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_RESIZE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_resize_context_t), sizeof(float));
}

int layer_cpu_float_RESIZE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_context_t* context = layer->C->context;
	layer_context_t* input_context = layer->inputs[0]->C->context;
	float* in = (float*)input_context->out[0];
	float* o = (float*)context->out[0];
	const int indices[1] = {0};
	const float boxes[] = {0, 0, 1, 1};

	r = CropAndResize_forward_cpu(o, in, boxes, indices, &context->nhwc, &input_context->nhwc);

	return r;
}
void layer_cpu_float_RESIZE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
