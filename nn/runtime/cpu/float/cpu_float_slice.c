/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_slice_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
int slice_ref(const float* IN, float* O, const NHWC_t* inhwc,
		const int32_t* begin, const int32_t* end, const int32_t* strides)
{
	int r = 0;
	int n,h,w,c;
	float* pO = O;

	assert(inhwc->N >= end[0]);
	assert(inhwc->H >= end[1]);
	assert(inhwc->W >= end[2]);
	assert(inhwc->C >= end[3]);

	for(n=begin[0]; n<end[0]; n+=strides[0]) {
		for(h=begin[1]; h<end[1]; h+=strides[1]) {
			for(w=begin[2]; w<end[2]; w+=strides[2]) {
				for(c=begin[3]; c<end[3]; c+=strides[3]) {
					*pO = IN[((n*inhwc->H+h)*inhwc->W+w)*inhwc->C+c];
					pO ++;
				}
			}
		}
	}

	if(pO == O) {
		r = NN_E_INVALID_LAYER;
	}

	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_SLICE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_slice_context_t), sizeof(float));
}

int layer_cpu_float_SLICE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	const layer_t* inL = layer->inputs[0];
	const layer_t* beginL = layer->inputs[1];
	const layer_t* sizeL = layer->inputs[2];
	const float* IN = (float*)inL->C->context->out[0];
	float* O = (float*)layer->C->context->out[0];
	const int32_t* p_begin = (int32_t*)beginL->C->context->out[0];
	const int32_t* p_size = (int32_t*)sizeL->C->context->out[0];
	int32_t begin[4];
	int32_t end[4];
	const int32_t strides[4] = { 1, 1, 1, 1 };

	assert(L_DT_FLOAT == inL->C->context->dtype);
	assert(L_DT_INT32 == beginL->C->context->dtype);
	assert(L_DT_INT32 == sizeL->C->context->dtype);
	if(3 == beginL->C->context->nhwc.C) {
		assert(3 == sizeL->C->context->nhwc.C);
		begin[0] = p_begin[0];
		begin[1] = p_begin[1];
		begin[2] = 0;
		begin[3] = p_begin[2];
		end[0] = p_begin[0] + p_size[0];
		end[1] = p_begin[1] + p_size[1];
		end[2] = 1;
		end[3] = p_begin[2] + p_size[2];
	} else if(4 == beginL->C->context->nhwc.C) {
		assert(4 == sizeL->C->context->nhwc.C);
		memcpy(begin, p_begin, sizeof(begin));
		end[0] = p_begin[0] + p_size[0];
		end[1] = p_begin[1] + p_size[1];
		end[2] = p_begin[2] + p_size[2];
		end[3] = p_begin[3] + p_size[3];
	} else {
		r = NN_E_NOT_SUPPORTED;
	}

	r = slice_ref(IN, O, &(inL->C->context->nhwc), begin, end, strides);

	return r;
}
void layer_cpu_float_SLICE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}


int layer_cpu_float_STRIDEDSLICE_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_slice_context_t), sizeof(float));
}

int layer_cpu_float_STRIDEDSLICE_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;

	const layer_t* inL = layer->inputs[0];
	const layer_t* beginL = layer->inputs[1];
	const layer_t* endL = layer->inputs[2];
	const layer_t* stridesL = layer->inputs[3];
	const float* IN = (float*)inL->C->context->out[0];
	float* O = (float*)layer->C->context->out[0];
	const int32_t* p_begin = (int32_t*)beginL->C->context->out[0];
	const int32_t* p_end = (int32_t*)endL->C->context->out[0];
	const int32_t* p_strides = (int32_t*)stridesL->C->context->out[0];

	int32_t begin[4];
	int32_t end[4];
	int32_t strides[4];
	const int* ldims = (int*)&(layer->C->context->nhwc);
	int32_t i;

	assert(L_DT_FLOAT == inL->C->context->dtype);
	assert(L_DT_INT32 == beginL->C->context->dtype);
	assert(L_DT_INT32 == endL->C->context->dtype);
	assert(L_DT_INT32 == stridesL->C->context->dtype);

	if(3 == beginL->C->context->nhwc.C) {
		assert(3 == endL->C->context->nhwc.C);
		assert(3 == stridesL->C->context->nhwc.C);
		begin[0] = p_begin[0];
		begin[1] = p_begin[1];
		begin[2] = 0;
		begin[3] = p_begin[2];
		end[0] = p_end[0];
		end[1] = p_end[1];
		end[2] = 1;
		end[3] = p_end[2];
		strides[0] = p_strides[0];
		strides[1] = p_strides[1];
		strides[2] = 1;
		strides[3] = p_strides[2];
	} else if(4 == beginL->C->context->nhwc.C) {
		assert(4 == endL->C->context->nhwc.C);
		assert(4 == stridesL->C->context->nhwc.C);
		memcpy(begin, p_begin, sizeof(begin));
		memcpy(end, p_end, sizeof(end));
		memcpy(strides, p_strides, sizeof(strides));
	} else {
		r = NN_E_NOT_SUPPORTED;
	}

	for(i=0; i<4; i++) {
		if((0 == end[i]) && (0 == begin[i])) {
			end[i] += ldims[i];
		}
	}

	r = slice_ref(IN, O, &(inL->C->context->nhwc), begin, end, strides);

	return r;
}
void layer_cpu_float_STRIDEDSLICE_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
