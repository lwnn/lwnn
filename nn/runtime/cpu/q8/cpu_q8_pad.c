/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q8
#include "../runtime_cpu.h"

#include "arm_math.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
} layer_cpu_q8_pad_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void padding_HWC_q7(const q7_t *Im_in,
	const uint16_t dim_im_in_x,
	const uint16_t dim_im_in_y,
	const uint16_t ch_im_in,
	const uint16_t padding_top,
	const uint16_t padding_bottom,
	const uint16_t padding_left,
	const uint16_t padding_right,
	q7_t *Im_out,
	const uint16_t dim_im_out_x,
	const uint16_t dim_im_out_y)
{
	int i, size;
	q7_t * p_out = Im_out;

	/* top rows */
	size = dim_im_out_x*ch_im_in*padding_top;
	memset(p_out, 0, size);
	p_out += size;

	/* middle */
	for(i=0; i<dim_im_in_y; i++)
	{
		/* left - set to 0 */
		size = ch_im_in * padding_left;
		memset(p_out, 0, size);
		p_out += size;
		/* data - copy a row */
		size = dim_im_in_x * ch_im_in;
		memcpy(p_out, Im_in + i*size, size);
		p_out += size;
		/* right - set to 0 */
		size = ch_im_in * padding_right;
		memset(p_out, 0, size);
		p_out += size;
	}
	/* bottom rows */
	memset(p_out, 0, dim_im_out_x*ch_im_in*padding_bottom);
}

/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_PAD_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_pad_context_t), sizeof(int8_t));
}

int layer_cpu_q8_PAD_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_q8_pad_context_t* context = (layer_cpu_q8_pad_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q8_context_t* input_context = (layer_cpu_q8_context_t*)input->C->context;

	int8_t* IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];

	size_t batch;
	size_t batch_sizeIn = NHWC_BATCH_SIZE(input_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);
	int * ints = layer->blobs[0]->blob;

	uint16_t padding_top = ints[1];
	uint16_t padding_bottom = ints[5];
	uint16_t padding_left = ints[2];
	uint16_t padding_right = ints[6];

	NNLOG(NN_DEBUG, ("execute %s: [%d %d %d %d]\n", layer->name,
			padding_top,padding_bottom, padding_left,padding_right));

	for(batch=0; (batch<input_context->nhwc.N) && (0 == r); batch++)
	{
		padding_HWC_q7(IN+batch_sizeIn*batch,
				input_context->nhwc.W,
				input_context->nhwc.H,
				input_context->nhwc.C,
				padding_top, padding_bottom,
				padding_left, padding_right,
				O+batch_sizeO*batch,
				context->nhwc.W,
				context->nhwc.H);
	}

	return r;
}

void layer_cpu_q8_PAD_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q8 */
