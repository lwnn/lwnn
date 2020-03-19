/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_pad_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void padding_HWC_ref(const float *Im_in,
	const int dim_im_in_x,
	const int dim_im_in_y,
	const int ch_im_in,
	const int padding_top,
	const int padding_bottom,
	const int padding_left,
	const int padding_right,
	float *Im_out,
	const int dim_im_out_x,
	const int dim_im_out_y,
	float value)
{
	int i, size;
	float * p_out = Im_out;

	/* top rows */
	size = dim_im_out_x*ch_im_in*padding_top;
	memset(p_out, value, size*sizeof(float));
	p_out += size;

	/* middle */
	for(i=0; i<dim_im_in_y; i++)
	{
		/* left - set to 0 */
		size = ch_im_in * padding_left;
		memset(p_out, value, size*sizeof(float));
		p_out += size;
		/* data - copy a row */
		size = dim_im_in_x * ch_im_in;
		memcpy(p_out, Im_in + i*size, size*sizeof(float));
		p_out += size;
		/* right - set to 0 */
		size = ch_im_in * padding_right;
		memset(p_out, value, size*sizeof(float));
		p_out += size;
	}
	/* bottom rows */
	size = dim_im_out_x*ch_im_in*padding_bottom;
	memset(p_out, value, size*sizeof(float));
}

/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_PAD_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_pad_context_t), sizeof(float));

}

int layer_cpu_float_PAD_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_pad_context_t* context = (layer_cpu_float_pad_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)input->C->context;

	float* IN = (float*)input_context->out[0];
	float *O = (float*)context->out[0];

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);
	int * ints = layer->blobs[0]->blob;

	int padding_top = ints[1];
	int padding_bottom = ints[5];
	int padding_left = ints[2];
	int padding_right = ints[6];

	float value = RTE_FETCH_FLOAT(layer->blobs[1]->blob, 0);

	assert(layer->blobs[0]->dims[0] == 8);
	NNLOG(NN_DEBUG, ("execute %s: [%d %d %d %d]\n", layer->name,
			padding_top,padding_bottom, padding_left,padding_right));

	for(batch=0; (batch<input_context->nhwc.N) && (0 == r); batch++)
	{
		padding_HWC_ref(IN+batch_sizeIn*batch,
				input_context->nhwc.W,
				input_context->nhwc.H,
				input_context->nhwc.C,
				padding_top, padding_bottom,
				padding_left, padding_right,
				O+batch_sizeO*batch,
				context->nhwc.W,
				context->nhwc.H,
				value);
	}

	return r;
}

void layer_cpu_float_PAD_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
