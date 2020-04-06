/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "runtime_halide.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	int dummy;
} rte_halide_t;
/* ============================ [ DECLARES  ] ====================================================== */
#define OP_DEF(op) L_OPS_DECLARE(halide_##op);
#include "opdef.h"
#undef OP_DEF
UNSUPPORTED_LAYER_OPS(halide, MAXIMUM)
UNSUPPORTED_LAYER_OPS(halide, RELU)
UNSUPPORTED_LAYER_OPS(halide, MAXPOOL)
UNSUPPORTED_LAYER_OPS(halide, RESHAPE)
UNSUPPORTED_LAYER_OPS(halide, DENSE)
UNSUPPORTED_LAYER_OPS(halide, SOFTMAX)
UNSUPPORTED_LAYER_OPS(halide, PAD)
UNSUPPORTED_LAYER_OPS(halide, DWCONV2D)
UNSUPPORTED_LAYER_OPS(halide, CONCAT)
UNSUPPORTED_LAYER_OPS(halide, AVGPOOL)
UNSUPPORTED_LAYER_OPS(halide, ADD)
UNSUPPORTED_LAYER_OPS(halide, CONST)
UNSUPPORTED_LAYER_OPS(halide, DETECTIONOUTPUT)
UNSUPPORTED_LAYER_OPS(halide, UPSAMPLE)
UNSUPPORTED_LAYER_OPS(halide, YOLO)
UNSUPPORTED_LAYER_OPS(halide, YOLOOUTPUT)
UNSUPPORTED_LAYER_OPS(halide, DECONV2D)
UNSUPPORTED_LAYER_OPS(halide, BATCHNORM)
UNSUPPORTED_LAYER_OPS(halide, DILCONV2D)
UNSUPPORTED_LAYER_OPS(halide, PRELU)
UNSUPPORTED_LAYER_OPS(halide, MFCC)
UNSUPPORTED_LAYER_OPS(halide, LSTM)
UNSUPPORTED_LAYER_OPS(halide, MINIMUM)
UNSUPPORTED_LAYER_OPS(halide, TRANSPOSE)
UNSUPPORTED_LAYER_OPS(halide, DETECTION)
UNSUPPORTED_LAYER_OPS(halide, PROPOSAL)
UNSUPPORTED_LAYER_OPS(halide, PYRAMID_ROI_ALIGN)
UNSUPPORTED_LAYER_OPS(halide, SLICE)
/* ============================ [ DATAS     ] ====================================================== */
static const layer_ops_t halide_lops[] =
{
#define OP_DEF(op) L_OPS_REF(halide_##op),
	#include "opdef.h"
#undef OP_DEF
};
/* ============================ [ LOCALS    ] ====================================================== */
static int halide_init_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;

	if(layer->op < ARRAY_SIZE(halide_lops))
	{
		r = halide_lops[layer->op].init(nn, layer);
	}

	return r;
}
static int halide_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;

	if(layer->op < ARRAY_SIZE(halide_lops))
	{
		NNLOG(NN_DEBUG, ("execute %s: [%d %d %d %d]\n", layer->name, L_SHAPES(layer)));
		r = halide_lops[layer->op].execute(nn, layer);
	}

	return r;
}

static int halide_deinit_layer(const nn_t* nn, const layer_t* layer)
{
	if(layer->op < ARRAY_SIZE(halide_lops))
	{
		halide_lops[layer->op].deinit(nn, layer);
	}

	return 0;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
extern "C" {
runtime_t rte_HALIDE_create(const nn_t* nn)
{
	rte_halide_t* rt = NULL;

	rt = (rte_halide_t*)malloc(sizeof(rte_halide_t));

	return rt;
}

int rte_HALIDE_init(const nn_t* nn)
{
	int r;
	rte_halide_t* rt = (rte_halide_t*)nn->runtime;

	r = rte_do_for_each_layer(nn, halide_init_layer);

	return r;
}

int rte_HALIDE_execute(const nn_t* nn)
{
	int r;
	rte_halide_t* rt = (rte_halide_t*)nn->runtime;

	r = rte_do_for_each_layer(nn, halide_execute_layer);

	return r;
}

void rte_HALIDE_destory(const nn_t* nn)
{
	rte_halide_t* rt = (rte_halide_t*)nn->runtime;

	rte_do_for_each_layer(nn, halide_deinit_layer);
}
}; /* extern "C"  end */


int rte_halide_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			size_t sz, size_t nout)
{
	int r = 0;
	layer_halide_context_t* context = NULL;
	rte_halide_t* rt = (rte_halide_t*)nn->runtime;

	assert(sz >= sizeof(layer_halide_context_t));

	context = (layer_halide_context_t*)malloc(sz+nout*sizeof(void*));

	if(context != NULL)
	{
		if(layer->dtype != L_DT_AUTO)
		{
			context->dtype = layer->dtype;
		}
		else
		{
			context->dtype = L_DT_FLOAT;
		}
		context->out = (void**)(((unsigned long long)context)+sz);
		context->nout = nout;
		if(nout > 0)
		{
			memset(context->out, 0, sizeof(void*)*nout);
		}
		r = layer_get_NHWC(layer, &context->nhwc);
		if(0 == r) {
			if(1 != context->nhwc.N) { /* only supprot batch 1 */
				r = NN_E_NOT_SUPPORTED;
			}
		}
		if(0 != r)
		{
			free(context);
		}
	}
	else
	{
		r = NN_E_NO_MEMORY;
	}

	if(0 == r)
	{
		layer->C->context = (layer_context_t*)context;
	}

	return r;
}

void rte_halide_destory_layer_context(const nn_t* nn, const layer_t* layer)
{
	size_t i;
	layer_halide_context_t* context = (layer_halide_context_t*)layer->C->context;

	if(NULL != context)
	{
		for(i=0; i< context->nout; i++) {
			delete (Halide::Func*)context->out[0];
		}
		free(context);
	}

	layer->C->context = NULL;
}

int rte_halide_create_layer_common(const nn_t* nn, const layer_t* layer, size_t ctx_sz)
{
	int r = 0;
	layer_halide_context_t* context;

	r = rte_halide_create_layer_context(nn, layer, ctx_sz, 1);

	if(0 == r)
	{
		context = (layer_halide_context_t*)layer->C->context;
	}

	context->out[0] = new Halide::Func(layer->name);
	if(NULL == context->out[0])
	{
		r = NN_E_NO_MEMORY;
		rte_halide_destory_layer_context(nn, layer);
	}
	return r;
}

Halide::Buffer<float>* rte_halide_create_buffer_from_blob(const nn_t* nn, const layer_blob_t* blob)
{
	Halide::Buffer<float>* buf = NULL;
	const float* data = (float*)blob->blob;
	int dim = 0;
	size_t size = 1;
	const int* dims = blob->dims;

	assert(dims != NULL);

	while((dims[dim] != 0) && (dim < 4)) {
		size = size*dims[dim];
		dim ++;
	};

	switch(dim) {
		case 1:
			buf = new Halide::Buffer<float>(dims[0]);
			break;
		case 2:
			buf = new Halide::Buffer<float>(dims[1], dims[0]);
			break;
		case 3:
			buf = new Halide::Buffer<float>(dims[2], dims[1], dims[0]);
			break;
		case 4:
			buf = new Halide::Buffer<float>(dims[3], dims[2], dims[1], dims[0]);
			break;
		default:
			break;
	}

	if(NULL != buf) {
		buf->allocate();
		std::copy(data, data+size, buf->begin());
	}

	return buf;
}


