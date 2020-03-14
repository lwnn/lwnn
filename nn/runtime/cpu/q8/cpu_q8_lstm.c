/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q8
#include "../runtime_cpu.h"

#include "arm_math.h"
#include "arm_common_tables.h"
#include "arm_nnfunctions.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_Q8_CONTEXT_MEMBER;
	int8_t* h;
	int8_t* c;
#if defined (ARM_MATH_DSP)
	rte_cpu_buffer_t* bufferA;
#endif
} layer_cpu_q8_lstm_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void concate_xh(int8_t* xh,int8_t*  x, int8_t Q, int8_t* h, int input_size, int output_size)
{
	int i;
	memcpy(xh, x, input_size);
	assert(7>=Q);
	if(7 > Q) {
		for(i=0; i<output_size; i++) {
			h[i] = h[i] >> (7-Q);
		}
	}
	memcpy(xh+input_size, h, output_size);
}
static void activation_q8_ref(int8_t * data, uint16_t size, int8_t Q, arm_nn_activation_type type)
{
	uint16_t  i = size;
	q7_t     *pIn = data;
	q7_t     *pOut = data;
	q7_t      in;
	q7_t      out;
	uint16_t  shift_size;
	int32_t eight;
	const q7_t *lookup_table;

	if((7-Q) <= 3) {
		arm_nn_activations_direct_q7(data, size, 7-Q, type);
	} else {
		switch (type) {
		case ARM_SIGMOID:
			lookup_table = sigmoidTable_q7;
			break;
		case ARM_TANH:
		default:
			lookup_table = tanhTable_q7;
			break;
		}
		if(Q>=0) {
			eight = 8<<Q;
		} else {
			eight = 8>>(-Q);
		}
		shift_size = 3-(7-Q);
		while (i) {
			in = *pIn++;
			if(in >= eight) {
				out = 0x7F; /* +1 */
			} else if((ARM_TANH == type) && (in <= -eight)) {
				out = 0x80; /* -1 */
			} else {
				if(shift_size > 0) {
					out = lookup_table[(uint8_t) (in >> shift_size)];
				} else {
					out = lookup_table[(uint8_t) (in << shift_size)];
				}
			}
			*pOut++ = out;
			i--;
		}
	}
}

static void cell_state_calc(int8_t* c, int8_t cQ, int8_t* ft, int8_t* it, int8_t* ct, int hidden_size)
{
	int i;
	int32_t out;

	for(i=0; i<hidden_size; i++){
		out = (((int32_t)ft[i]*c[i])>>7) + (((int32_t)it[i]*ct[i]) >> (14-cQ));
		c[i] = (int8_t) __SSAT(out, 8);
	}
}

static void output_calc(int8_t* h, int8_t* Ct, int8_t cQ, int8_t* ot, int hidden_size)
{
	int i;
	int32_t out;
	memcpy(h, Ct, hidden_size);
	activation_q8_ref(h, hidden_size, cQ, ARM_TANH);
	for(i=0; i<hidden_size; i++){
		out =(int32_t)ot[i]*h[i] + NN_ROUND(7);
		h[i] =(int8_t) __SSAT(out>>7, 8);
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_LSTM_init(const nn_t* nn, const layer_t* layer)
{
	int hidden_size;
	int input_size;
	int output_size;
	int num_directions;
	layer_cpu_q8_lstm_context_t* context;
	int r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_q8_lstm_context_t), sizeof(int8_t));

	if(0 == r) {
		context = (layer_cpu_q8_lstm_context_t*)layer->C->context;
		num_directions = layer->blobs[1]->dims[0];
		hidden_size = layer->blobs[1]->dims[1]/4;
		output_size = context->nhwc.C;
		input_size = layer->inputs[0]->C->context->nhwc.C;
		context->c = malloc(num_directions*sizeof(int8_t)*(hidden_size+output_size));
		context->h = context->c + num_directions*hidden_size;
		nn_request_scratch(nn, sizeof(int8_t)*(4*hidden_size + (input_size+output_size)));
		if(NULL == context->c) {
			r = NN_E_NO_MEMORY;
		} else {
			memset(context->c, 0, num_directions*sizeof(int8_t)*(hidden_size+output_size));
		}
	}

#if defined (ARM_MATH_DSP)
	if(0 == r)
	{
		context->bufferA = rte_cpu_create_buffer(nn, layer, RTE_FETCH_INT32(layer->blobs[0]->dims, 0)*sizeof(q15_t));

		if(NULL == context->bufferA)
		{
			r = NN_E_NO_MEMORY;
		}
		else
		{
			rte_cpu_release_buffer(context->bufferA);
		}
	}
#endif

	return r;
}

int layer_cpu_q8_LSTM_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int batch_size, input_size, hidden_size, output_size, i;
	layer_cpu_q8_lstm_context_t* context = (layer_cpu_q8_lstm_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	layer_cpu_q8_context_t* input_context = (layer_cpu_q8_context_t*)input->C->context;
	int8_t *IN = (int8_t*)input_context->out[0];
	int8_t *O = (int8_t*)context->out[0];
	const int8_t *W = (const int8_t*)layer->blobs[1]->blob;
	const int8_t *B = (const int8_t*)layer->blobs[2]->blob;
	int8_t wQ, bQ, cQ, gQ;
	int8_t *x, *y, *c, *h, *xh;
	int8_t *gates,*it,*ft,*ct,*ot;
	int8_t* ints = (int8_t*)layer->blobs[3]->blob;
	int num_directions, d;

	wQ = ints[0];
	bQ = ints[1];
	cQ = ints[2];
	gQ = ints[3];

	batch_size = input_context->nhwc.H;
	input_size = input_context->nhwc.C;
	output_size = context->nhwc.C;
	hidden_size = layer->blobs[1]->dims[1]/4;;
	num_directions = layer->blobs[1]->dims[0];

	NNLOG(NN_DEBUG, ("execute %s: B=%d, I=%d, H=%d, O=%d, D=%d, %dx%d+%d(%d,%d) -> %d\n",
			layer->name, batch_size, input_size, hidden_size, output_size, num_directions,
			LAYER_Q(input), wQ, bQ, cQ, gQ, LAYER_Q(layer)));

	gates = (int8_t*)nn->scratch.area;
	it = gates;
	ot = gates + hidden_size;
	ft = ot + hidden_size;
	ct = ft + hidden_size;
	xh = ct + hidden_size;

	for(d=0; (d<num_directions) && (0==r); d++) {
		c = context->c + d*hidden_size;
		h = context->h + d*output_size;
		x = (int8_t*)input_context->out[0];
		y = (int8_t*)context->out[0];
		y += d*output_size;

		W = (const int8_t*) layer->blobs[1]->blob;
		W += d*hidden_size*(input_size+output_size)*4;
		B = (const int8_t*) layer->blobs[2]->blob;
		B += d*hidden_size*8;

		assert(wQ+LAYER_Q(input)-bQ>=0);
		assert(wQ+LAYER_Q(input)-gQ>=0);
		assert(7==LAYER_Q(layer));

		for(i=0; (i<batch_size) && (0==r); i++) {
			concate_xh(xh, x, LAYER_Q(input), h, input_size, output_size);
			r = arm_fully_connected_q7_opt(xh,
					W,
					(input_size+output_size),
					hidden_size*4,
					wQ+LAYER_Q(input)-bQ,
					wQ+LAYER_Q(input)-gQ,
					B,
					gates,
					#if defined (ARM_MATH_DSP)
					context->bufferA->data
					#else
					NULL
					#endif
					);
			activation_q8_ref(it, 3*hidden_size, gQ, ARM_SIGMOID);
			activation_q8_ref(ct, hidden_size, gQ, ARM_TANH);
			cell_state_calc(c, cQ, ft, it, ct, hidden_size);
			output_calc(h, c, cQ, ot, hidden_size);
			memcpy(y, h, output_size);
			if(context->nhwc.H == batch_size) {
				y = y + num_directions*output_size;
			}
			x += input_size;
		}
	}
	return r;
}

void layer_cpu_q8_LSTM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_q8_lstm_context_t* context = (layer_cpu_q8_lstm_context_t*)layer->C->context;

	if(NULL != context) {
		if(NULL != context->c) free(context->c);
	}

	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_Q8 */
