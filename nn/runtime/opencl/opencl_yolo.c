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
	void* pout;
} layer_cl_yolo_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_YOLO_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	layer_cl_yolo_context_t* context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context = (layer_cl_context_t*)input->C->context;

	r = rte_cl_create_layer_context(nn, layer, NULL, NULL, NULL, sizeof(layer_cl_yolo_context_t), 0);

	if(0 == r)
	{
		context = (layer_cl_yolo_context_t*) layer->C->context;
		context->pout = malloc(NHWC_SIZE(context->nhwc)*sizeof(float));

		if(NULL == context->pout)
		{
			rte_cl_destory_layer_context(nn, layer);
			r = NN_E_NO_MEMORY;
		} else {
			nn_request_scratch(nn, NHWC_SIZE(input_context->nhwc)*sizeof(float));
		}
	}

	return r;
}

int layer_cl_YOLO_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cl_yolo_context_t* context = (layer_cl_yolo_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cl_context_t* input_context = (layer_cl_context_t*)input->C->context;
	int num = layer->blobs[0]->dims[0];
	int classes = RTE_FETCH_FLOAT(layer->blobs[2]->blob, 0);

	float* in = (float*)nn->scratch.area;

	NNLOG(NN_DEBUG, ("execute %s\n",layer->name));

	r = rte_cl_image2d_copy_out(nn, input_context->out[0], in, &input_context->nhwc);

	if(0 == r)
	{
		r = yolo_forward(context->pout, in, &input_context->nhwc, num, classes);
	}

	return r;
}

void layer_cl_YOLO_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cl_yolo_context_t* context;
	context = (layer_cl_yolo_context_t*) layer->C->context;
	if(NULL != context)
	{
		free(context->pout);
		rte_cl_destory_layer_context(nn, layer);
	}

}

#endif /* DISABLE_RUNTIME_CL */
