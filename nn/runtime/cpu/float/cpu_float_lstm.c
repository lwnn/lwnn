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
		const float* Wb, const float* Rb, float* gate,
		int input_size, int hidden_size, layer_activation_type_t activation)
{
	int i,j;
	float o;

	for(i=0; i<hidden_size; i++) {
		o = Wb[i] + Rb[i];
		for(j=0; j<input_size; j++) {
			o += x[j]*W[i*input_size+j];
		}
		for(j=0; j<hidden_size; j++) {
			o += h[j]*R[i*hidden_size+j];
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
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_LSTM_init(const nn_t* nn, const layer_t* layer)
{
	int hidden_size;
	layer_cpu_float_lstm_context_t* context;
	int r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_lstm_context_t), sizeof(float));

	if(0 == r) {
		context = (layer_cpu_float_lstm_context_t*)layer->C->context;
		hidden_size = layer->blobs[1]->dims[2];
		context->c = malloc(sizeof(float)*hidden_size*2);
		context->h = context->c + hidden_size;
		nn_request_scratch(nn, 3*sizeof(float)*hidden_size);
		if(NULL == context->c) {
			r = NN_E_NO_MEMORY;
		} else {
			memset(context->c, 0, sizeof(float)*hidden_size*2);
		}
	}

	return r;
}
int layer_cpu_float_LSTM_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int batch_size, input_size, hidden_size, i;
	layer_cpu_float_lstm_context_t* context = (layer_cpu_float_lstm_context_t*)layer->C->context;
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)layer->inputs[0]->C->context;
	float* x = (float*)input_context->out[0];
	float* y = (float*)context->out[0];
	const float *Wi,*Wo,*Wf,*Wc;
	const float *Ri,*Ro,*Rf,*Rc;
	const float *Wbi,*Wbo,*Wbf,*Wbc;
	const float *Rbi,*Rbo,*Rbf,*Rbc;
	float *it,*ft,*ct,*ot;
	float* h = context->h;

	batch_size = input_context->nhwc.H;
	assert(batch_size == context->nhwc.H);
	input_size = input_context->nhwc.C;
	assert(input_size==layer->blobs[0]->dims[2]);
	hidden_size = layer->blobs[1]->dims[2];

	Wi = (const float*) layer->blobs[0]->blob;
	Wo = Wi + hidden_size*input_size;
	Wf = Wo + hidden_size*input_size;
	Wc = Wf + hidden_size*input_size;

	Ri = (const float*) layer->blobs[1]->blob;
	Ro = Ri + hidden_size*hidden_size;
	Rf = Ro + hidden_size*hidden_size;
	Rc = Rf + hidden_size*hidden_size;

	Wbi = (const float*) layer->blobs[2]->blob;
	Wbo = Wbi + hidden_size;
	Wbf = Wbo + hidden_size;
	Wbc = Wbf + hidden_size;

	Rbi = Wbc + hidden_size;
	Rbo = Rbi + hidden_size;
	Rbf = Rbo + hidden_size;
	Rbc = Rbf + hidden_size;

	it = (float*)nn->scratch.area;
	ot = it;
	ft = ot + hidden_size;
	ct = ft + hidden_size;

	NNLOG(NN_DEBUG, ("execute %s: B=%d, I=%d, H=%d\n", layer->name, batch_size, input_size, hidden_size));

	for(i=0; i<batch_size; i++) {
		gate_calc(x, Wi, h, Ri, Wbi, Rbi, it, input_size, hidden_size, L_ACT_SIGMOID);
		gate_calc(x, Wf, h, Rf, Wbf, Rbf, ft, input_size, hidden_size, L_ACT_SIGMOID);
		gate_calc(x, Wc, h, Rc, Wbc, Rbc, ct, input_size, hidden_size, L_ACT_TANH);
		cell_state_calc(context->c, ft, it, ct, hidden_size);
		gate_calc(x, Wo, h, Ro, Wbo, Rbo, ot, input_size, hidden_size, L_ACT_SIGMOID);
		output_calc(h, context->c, ot, hidden_size);
		memcpy(y, h, hidden_size*sizeof(float));
		if(context->nhwc.H == batch_size) {
			y = y + hidden_size;
		}
		x += input_size;
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
