/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
} layer_cl_detection_output_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_DETECTIONOUTPUT_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_create_layer_common(nn, layer, NULL, NULL, sizeof(layer_cl_detection_output_context_t));
}

void layer_cl_DETECTIONOUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
