/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
#include <math.h>
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
	float* h;
	float* c;
} layer_cpu_float_lstm_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void gate_calc(float* x, const float* W, float* h, const float* R,
		const float* Wb, const float* Rb, const float* P, const float* c, float* gate,
		int input_size, int hidden_size, int output_size, layer_activation_type_t activation)
{
	int i,j;
	float o;

	for(i=0; i<hidden_size; i++) {
		o = Wb[i] + Rb[i];
		for(j=0; j<input_size; j++) {
			o += x[j]*W[i*input_size+j];
		}

		for(j=0; j<output_size; j++) {
			o += h[j]*R[i*output_size+j];
		}

		if(NULL != P) {
			o += P[i]*c[i];
		}

		switch(activation){
		case L_ACT_SIGMOID:
			o = 1 / (1 + exp(-o));
			break;
		case L_ACT_TANH:
			o = tanh(o);
			break;
		default:
			break;
		}
		gate[i] = o;
	}
}

static void cell_state_calc(float* c, float* ft, float* it, float* ct, int hidden_size)
{
	int i;

	for(i=0; i<hidden_size; i++){
		c[i] = ft[i]*c[i] + it[i]*ct[i];
	}
}

static void output_calc(float* h, float* Ct, float* ot, int hidden_size)
{
	int i;

	for(i=0; i<hidden_size; i++){
		h[i] = ot[i]*tanh(Ct[i]);
	}
}

static void projection(float* h, float* ht, const float* PJ, int hidden_size, int output_size) {
	int i,j;
	float o;
	for(i=0; i<output_size; i++) {
		o = 0;
		for(j=0; j<hidden_size; j++) {
			o += ht[j]*PJ[i*hidden_size+j];
		}
		h[i] = o;
	}
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_LSTM_init(const nn_t* nn, const layer_t* layer)
{
	int hidden_size;
	int output_size;
	int num_directions;
	size_t scratch_size;
	layer_cpu_float_lstm_context_t* context;
	int r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_lstm_context_t), sizeof(float));

	if(0 == r) {
		context = (layer_cpu_float_lstm_context_t*)layer->C->context;
		num_directions = layer->blobs[0]->dims[0];
		hidden_size = layer->blobs[0]->dims[1]/4;
		output_size = context->nhwc.C;
		context->c = malloc(num_directions*sizeof(float)*(hidden_size+output_size));
		context->h = context->c + num_directions*hidden_size;
		scratch_size = 3*sizeof(float)*hidden_size;
		#if !defined(DISABLE_RTE_FALLBACK) && !defined(DISABLE_RUNTIME_OPENCL)
		if(RUNTIME_OPENCL == nn->runtime_type) { /* those are used for fallback */
			scratch_size += sizeof(float)*NHWC_SIZE(layer->inputs[0]->C->context->nhwc) + sizeof(void*);
			scratch_size += sizeof(float)*NHWC_SIZE(context->nhwc) + sizeof(void*);
		}
		#endif
		nn_request_scratch(nn, scratch_size);
		if(NULL == context->c) {
			r = NN_E_NO_MEMORY;
		} else {
			memset(context->c, 0, num_directions*sizeof(float)*(hidden_size+output_size));
		}
	}

	return r;
}
int layer_cpu_float_LSTM_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int batch_size, input_size, hidden_size, output_size, i;
	layer_cpu_float_lstm_context_t* context = (layer_cpu_float_lstm_context_t*)layer->C->context;
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)layer->inputs[0]->C->context;
	float* x;
	float* y;
	const float *Wi,*Wo,*Wf,*Wc;
	const float *Ri,*Ro,*Rf,*Rc;
	const float *Wbi,*Wbo,*Wbf,*Wbc;
	const float *Rbi,*Rbo,*Rbf,*Rbc;
	const float *Pi=NULL, *Pf=NULL, *Po=NULL;
	const float *PJ = NULL;
	float *it,*ft,*ct,*ot;
	float* c;
	float* h;
	const int* ints;
	int num_directions, d;

	batch_size = input_context->nhwc.H;
	input_size = input_context->nhwc.C;
	output_size = layer->blobs[1]->dims[2];
	assert(input_size==layer->blobs[0]->dims[2]);
	hidden_size = layer->blobs[0]->dims[1]/4;;
	num_directions = layer->blobs[0]->dims[0];

	NNLOG(NN_DEBUG, ("execute %s: B=%d, I=%d, H=%d, O=%d, D=%d\n", layer->name,
			batch_size, input_size, hidden_size, output_size, num_directions));

	it = (float*)nn->scratch.area;
	#if !defined(DISABLE_RTE_FALLBACK) && !defined(DISABLE_RUNTIME_OPENCL)
	if(RUNTIME_OPENCL == nn->runtime_type) { /* skip those which are used for fallback */
		it = (float*)(((size_t)it)
				+ sizeof(float)*NHWC_SIZE(input_context->nhwc) + sizeof(void*)
				+ sizeof(float)*NHWC_SIZE(context->nhwc) + sizeof(void*));
	}
	#endif
	ot = it;
	ft = ot + hidden_size;
	ct = ft + hidden_size;

	for(d=0; d<num_directions; d++) {
		c = context->c + d*hidden_size;
		h = context->h + d*output_size;
		x = (float*)input_context->out[0];
		y = (float*)context->out[0];
		y += d*output_size;

		Wi = (const float*) layer->blobs[0]->blob;
		Wi += d*hidden_size*input_size*4;
		Wo = Wi + hidden_size*input_size;
		Wf = Wo + hidden_size*input_size;
		Wc = Wf + hidden_size*input_size;

		Ri = (const float*) layer->blobs[1]->blob;
		Ri += d*hidden_size*output_size*4;
		Ro = Ri + hidden_size*output_size;
		Rf = Ro + hidden_size*output_size;
		Rc = Rf + hidden_size*output_size;

		Wbi = (const float*) layer->blobs[2]->blob;
		Wbi += d*hidden_size*8;
		Wbo = Wbi + hidden_size;
		Wbf = Wbo + hidden_size;
		Wbc = Wbf + hidden_size;

		Rbi = Wbc + hidden_size;
		Rbo = Rbi + hidden_size;
		Rbf = Rbo + hidden_size;
		Rbc = Rbf + hidden_size;

		if(NULL != layer->blobs[3]) {
			ints = (const int*)layer->blobs[3]->blob;
			for(i=0; i<layer->blobs[3]->dims[0]; i++) {
				switch (ints[i]) {
				case 0: /* peephole */
					Pi = (const float*)layer->blobs[4+i]->blob;
					Pi += d*hidden_size*3;
					Pf = Pi + hidden_size;
					Po = Pf + hidden_size;
					break;
				case 1: /* projection */
					PJ = (const float*)layer->blobs[4+i]->blob;
					PJ += d*hidden_size*output_size;
					break;
				}
			}
		}

		for(i=0; i<batch_size; i++) {
			gate_calc(x, Wi, h, Ri, Wbi, Rbi, Pi, c, it, input_size, hidden_size, output_size, L_ACT_SIGMOID);
			gate_calc(x, Wf, h, Rf, Wbf, Rbf, Pf, c, ft, input_size, hidden_size, output_size, L_ACT_SIGMOID);
			gate_calc(x, Wc, h, Rc, Wbc, Rbc, NULL, NULL, ct, input_size, hidden_size, output_size, L_ACT_TANH);
			cell_state_calc(c, ft, it, ct, hidden_size);
			gate_calc(x, Wo, h, Ro, Wbo, Rbo, Po, c, ot, input_size, hidden_size, output_size, L_ACT_SIGMOID);
			if(NULL != PJ) {
				output_calc(ct, c, ot, hidden_size);
				projection(h, ct, PJ, hidden_size, output_size);
			} else {
				output_calc(h, c, ot, hidden_size);
			}
			memcpy(y, h, output_size*sizeof(float));
			if(context->nhwc.H == batch_size) {
				y = y + num_directions*output_size;
			}
			x += input_size;
		}
	}

	return r;
}
void layer_cpu_float_LSTM_deinit(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_lstm_context_t* context = (layer_cpu_float_lstm_context_t*)layer->C->context;

	if(NULL != context) {
		if(NULL != context->c) free(context->c);
	}

	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
