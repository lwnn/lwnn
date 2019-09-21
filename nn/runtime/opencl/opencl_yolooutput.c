/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
#include "yolo.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
} layer_cl_yolooutput_context_t;

typedef struct {
	LAYER_CL_CONTEXT_MEMBER;
	void* pout;
} layer_cl_yolo_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void* fetch_yolo_out(const nn_t* nn, const layer_t* layer)
{
	layer_cl_yolo_context_t* context;
	context = (layer_cl_yolo_context_t*)layer->C->context;
	(void)nn;
	NNLOG(NN_DEBUG, ("  cl fetch %s\n",layer->name));
	return context->pout;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_YOLOOUTPUT_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_create_layer_context(nn, layer, NULL, NULL, NULL, sizeof(layer_cl_yolooutput_context_t), 0);
}

int layer_cl_YOLOOUTPUT_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_yolooutput_context_t* context = (layer_cl_yolooutput_context_t*)layer->C->context;

	NNLOG(NN_DEBUG, ("execute %s\n",layer->name));

	r = yolo_output_forward(nn, layer, fetch_yolo_out);

	return r;
}

void layer_cl_YOLOOUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
