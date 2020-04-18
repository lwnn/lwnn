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
} layer_cpu_float_yolooutput_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_YOLOOUTPUT_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_float_yolooutput_context_t), 0);
}

int layer_cpu_float_YOLOOUTPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_yolooutput_context_t* context = (layer_cpu_float_yolooutput_context_t*)layer->C->context;

	r = yolo_output_forward(nn, layer);

	return r;
}

void layer_cpu_float_YOLOOUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
