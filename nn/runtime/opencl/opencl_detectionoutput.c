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
	int r;
	layer_cl_context_t* context;
	layer_cl_context_t* mbox_loc_context;
	layer_cl_context_t* mbox_conf_context;
	size_t scratch_size;

	r = rte_cl_create_layer_context(nn, layer, NULL, NULL, sizeof(layer_cl_detection_output_context_t), 0);

	if(0 == r)
	{
		context = (layer_cl_context_t*) layer->C->context;
		mbox_loc_context =
				(layer_cl_context_t*) layer->inputs[0]->C->context;
		mbox_conf_context =
				(layer_cl_context_t*) layer->inputs[1]->C->context;

		scratch_size = NHWC_SIZE(mbox_loc_context->nhwc) + NHWC_SIZE(mbox_conf_context->nhwc);
		nn_request_scratch(nn, scratch_size*sizeof(float));
	}

	return r;
}

void layer_cl_DETECTIONOUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cl_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_OPENCL */
