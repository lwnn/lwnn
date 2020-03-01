/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU
#include "runtime_cpu.h"
#ifndef DISABLE_RTE_FALLBACK
#include "quantize.h"
#endif
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	STAILQ_HEAD(rte_cpu_buffer_head,rte_cpu_buffer) buffers;
} rte_cpu_t;
/* ============================ [ DECLARES  ] ====================================================== */
#ifndef DISABLE_RUNTIME_CPU_Q8
#define OP_DEF(op) L_OPS_DECLARE(cpu_q8_##op);
#include "opdef.h"
#undef OP_DEF
#endif
#ifndef DISABLE_RUNTIME_CPU_S8
#define OP_DEF(op) L_OPS_DECLARE(cpu_s8_##op);
#include "opdef.h"
#undef OP_DEF
#endif
#ifndef DISABLE_RUNTIME_CPU_Q16
#define OP_DEF(op) L_OPS_DECLARE(cpu_q16_##op);
#include "opdef.h"
#undef OP_DEF
#endif
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#define OP_DEF(op) L_OPS_DECLARE(cpu_float_##op);
#include "opdef.h"
#undef OP_DEF
#endif

#ifndef DISABLE_NN_DDO
extern void rte_ddo_save(const nn_t* nn, const layer_t* layer);
#endif
/* ============================ [ DATAS     ] ====================================================== */
static const layer_ops_t cpu_lops[][L_OP_NUMBER] =
{
#ifndef DISABLE_RUNTIME_CPU_Q8
	{
		#define OP_DEF(op) L_OPS_REF(cpu_q8_##op),
		#include "opdef.h"
		#undef OP_DEF
	},
#endif
#ifndef DISABLE_RUNTIME_CPU_S8
	{
		#define OP_DEF(op) L_OPS_REF(cpu_s8_##op),
		#include "opdef.h"
		#undef OP_DEF
	},
#endif
#ifndef DISABLE_RUNTIME_CPU_Q16
	{
		#define OP_DEF(op) L_OPS_REF(cpu_q16_##op),
		#include "opdef.h"
		#undef OP_DEF
	},
#endif
#ifndef DISABLE_RUNTIME_CPU_FLOAT
	{
		#define OP_DEF(op) L_OPS_REF(cpu_float_##op),
		#include "opdef.h"
		#undef OP_DEF
	},
#endif
};
/* ============================ [ LOCALS    ] ====================================================== */
#ifndef DISABLE_NN_LOG
static int cpu_get_buffer_id(const nn_t* nn, rte_cpu_buffer_t* buffer)
{
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;
	int bufferId = -1;
	int id = -1;
	rte_cpu_buffer_t* b;

	STAILQ_FOREACH(b, &(rt->buffers), entry)
	{
		id ++;
		if(b == buffer)
		{
			bufferId = id;
			break;
		}
	}

	return bufferId;
}
#endif

static int cpu_init_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	if(nn->network->type < ARRAY_SIZE(cpu_lops))
	{
		if(layer->op < ARRAY_SIZE(cpu_lops[nn->network->type]))
		{
			r = cpu_lops[nn->network->type][layer->op].init(nn, layer);
		}
	}
	else
	{
		r = NN_E_INVALID_RUNTIME;
	}

	return r;
}

static int cpu_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	if(nn->network->type < ARRAY_SIZE(cpu_lops))
	{
		if(layer->op < ARRAY_SIZE(cpu_lops[nn->network->type]))
		{
			r = cpu_lops[nn->network->type][layer->op].execute(nn, layer);
#ifndef DISABLE_NN_DDO
			NNDDO(NN_DEBUG, rte_ddo_save(nn, layer));
#endif
		}
	}
	else
	{
		r = NN_E_INVALID_RUNTIME;
	}

	return r;
}

static int cpu_deinit_layer(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	if(nn->network->type < ARRAY_SIZE(cpu_lops))
	{
		if(layer->op < ARRAY_SIZE(cpu_lops[nn->network->type]))
		{
			cpu_lops[nn->network->type][layer->op].deinit(nn, layer);
		}
	}

	return 0;
}

static int cpu_adjust_layer_buffer(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int i;
	rte_cpu_buffer_t* buffer;
	layer_cpu_context_t* context = (layer_cpu_context_t*)layer->C->context;

	for(i=0; i<context->nout; i++)
	{
		buffer = context->out[i];
		if(NULL != buffer)
		{
			context->out[i] = buffer->data;
			NNLOG(NN_DEBUG, (" layer %s out[%d] using buffer%d\n", layer->name, i, cpu_get_buffer_id(nn, buffer)));
		}
	}

	return r;
}

/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t rte_CPU_create(const nn_t* nn)
{
	return malloc(sizeof(rte_cpu_t));
}

void rte_CPU_destory(const nn_t* nn)
{
	rte_cpu_buffer_t* b;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	rte_do_for_each_layer(nn, cpu_deinit_layer);

	while(FALSE == STAILQ_EMPTY(&rt->buffers))
	{
		b = STAILQ_FIRST(&rt->buffers);
		STAILQ_REMOVE_HEAD(&rt->buffers, entry);
		if(b->data != NULL)
		{
			free(b->data);
		}
		free(b);
	}

	free(nn->runtime);
}

int rte_CPU_init(const nn_t* nn)
{
	int r;
	rte_cpu_buffer_t* b;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;
#ifndef DISABLE_NN_LOG
	size_t sum = 0;
	int bufferId = -1;
#endif

	STAILQ_INIT(&(rt->buffers));

	r = rte_do_for_each_layer(nn, cpu_init_layer);

	if(0 == r)
	{
		NNLOG(NN_DEBUG, ("Memory Usage:\n"));
		STAILQ_FOREACH(b, &(rt->buffers), entry)
		{
			b->data = malloc(b->sz);
			if(b->data == NULL)
			{
				r = NN_E_NO_MEMORY;
				break;
			}

			#ifndef DISABLE_NN_LOG
			sum += b->sz;
			bufferId ++;
			#endif
			NNLOG(NN_DEBUG, (" buffer%d: %d\n", bufferId, (int)b->sz));
		}
		NNLOG(NN_DEBUG, (" summary: %d\n", (int)sum));
	}

	if(0 == r)
	{
		r = rte_do_for_each_layer(nn, cpu_adjust_layer_buffer);
	}

	return r;
}

int rte_CPU_execute(const nn_t* nn)
{
	return rte_do_for_each_layer(nn, cpu_execute_layer);
}

int rte_cpu_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			size_t sz, size_t nout)
{
	int r = 0;
	layer_cpu_context_t* context = NULL;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	assert(sz >= sizeof(layer_cpu_context_t));

	context = malloc(sz+nout*sizeof(void*));

	if(context != NULL)
	{
		switch(nn->network->type)
		{
			#ifndef DISABLE_RUNTIME_CPU_Q8
			case NETWORK_TYPE_Q8:
				context->dtype = L_DT_INT8;
				break;
			#endif
			#ifndef DISABLE_RUNTIME_CPU_S8
			case NETWORK_TYPE_S8:
				context->dtype = L_DT_INT8;
				break;
			#endif
			#ifndef DISABLE_RUNTIME_CPU_Q16
			case NETWORK_TYPE_Q16:
				context->dtype = L_DT_INT16;
				break;
			#endif
			#ifndef DISABLE_RUNTIME_CPU_FLOAT
			case NETWORK_TYPE_FLOAT:
				context->dtype = L_DT_FLOAT;
				break;
			#endif
			default:
				assert(0);
				break;
		}
		if(layer->dtype != L_DT_AUTO)
		{
			context->dtype = layer->dtype;
		}
		context->out = (void**)(((unsigned long long)context)+sz);
		context->nout = nout;
		if(nout > 0)
		{
			memset(context->out, 0, sizeof(void*)*nout);
		}
		r = layer_get_NHWC(layer, &context->nhwc);
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

		RTE_CPU_LOG_LAYER_SHAPE(layer);
	}

	return r;
}

void rte_cpu_destory_layer_context(const nn_t* nn, const layer_t* layer)
{
	size_t i;
	layer_cpu_context_t* context = (layer_cpu_context_t*)layer->C->context;

	if(NULL != context)
	{
		free(context);
	}

	layer->C->context = NULL;
}

int rte_cpu_create_layer_common(const nn_t* nn, const layer_t* layer, size_t ctx_sz, size_t type_sz)
{
	int r = 0;
	int bsz = 0;
	layer_cpu_context_t* context;

	r = rte_cpu_create_layer_context(nn, layer, ctx_sz, 1);

	if(0 == r)
	{
		context = (layer_cpu_context_t*)layer->C->context;
		bsz = NHWC_SIZE(context->nhwc)*type_sz;
	}

	if(bsz > 0)
	{
		#ifndef DISABLE_RTE_FALLBACK
		if(RUNTIME_CPU != nn->runtime_type)
		{
			context->out[0] = malloc(bsz);
		}
		else
		{
		#endif
			context->out[0] = rte_cpu_create_buffer(nn, layer, bsz);
		#ifndef DISABLE_RTE_FALLBACK
		}
		#endif
		if(NULL == context->out[0])
		{
			r = NN_E_NO_MEMORY;
			rte_cpu_destory_layer_context(nn, layer);
		}
	}

	return r;
}


void* rte_cpu_create_buffer(const nn_t* nn, const layer_t* layer, size_t sz)
{
	int r;
	rte_cpu_buffer_t* buffer = NULL;
	rte_cpu_buffer_t* b;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	STAILQ_FOREACH(b, &(rt->buffers), entry)
	{
		if(NULL == b->owner)
		{
			buffer = b;
			break;
		}
		else
		{
			r = rte_is_layer_consumed_from(nn, b->owner, layer);
			if(FALSE == r)
			{
				buffer = b;
				break;
			}
		}
	}

	if(NULL == buffer)
	{
		buffer = malloc(sizeof(rte_cpu_buffer_t));
		if(NULL != buffer)
		{
			buffer->owner = layer;
			buffer->sz = sz;
			buffer->data = NULL;

			STAILQ_INSERT_TAIL(&(rt->buffers), buffer, entry);
		}
	}
	else
	{
		buffer->owner = layer;
		if(sz > buffer->sz)
		{
			buffer->sz = sz;
		}
	}

	return buffer;
}

void rte_cpu_take_buffer(rte_cpu_buffer_t* buffer, const layer_t* layer)
{
	assert(buffer != NULL);
	buffer->owner = layer;
}

void rte_cpu_release_buffer(rte_cpu_buffer_t* buffer)
{
	assert(buffer != NULL);
	buffer->owner = NULL;
}


void* rte_cpu_fetch_out0(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_context_t* context;
	context = (layer_cpu_context_t*)layer->C->context;
	(void)nn;

	return context->out[0];
}

#ifndef DISABLE_RTE_FALLBACK
void rte_cpuq_to_cpu_float_init_common(const nn_t* nn, const layer_t* layer)
{
	size_t scratch_size=0;
	layer_cpu_context_t* context;
	const layer_t* const* inputs;

	if(L_OP_YOLOOUTPUT == layer->op)
	{
		return;
	}

	inputs = layer->inputs;
	while(NULL != (*inputs))
	{
		context = (layer_cpu_context_t*)(*inputs)->C->context;
		scratch_size += sizeof(float)*NHWC_SIZE(context->nhwc) + sizeof(void*);
		inputs++;
	}

	nn_request_scratch(nn, scratch_size);
}

int rte_cpuq_to_cpu_float_pre_execute_common(const nn_t* nn, const layer_t* layer)
{
	int r=0;
	layer_cpu_context_t* context;
	const layer_t* const* inputs;
	void** cl_inputs = (void**)nn->scratch.area;
	float* pf;

	if(L_OP_YOLOOUTPUT == layer->op)
	{
		return r;
	}

	inputs = layer->inputs;
	while(NULL != (*inputs))
	{
		context = (layer_cpu_context_t*)(*inputs)->C->context;
		*cl_inputs++ = context->out[0];
		inputs++;
	}

	pf = (float*)cl_inputs;

	inputs = layer->inputs;
	while((NULL != (*inputs)) && (0 == r))
	{
		context = (layer_cpu_context_t*) (*inputs)->C->context;
		switch(nn->network->type)
		{
			#if !defined(DISABLE_RUNTIME_CPU_Q8)
			case NETWORK_TYPE_Q8:
				dequantize_q8(pf, (int8_t*)context->out[0], NHWC_SIZE(context->nhwc),
						LAYER_Q((*inputs)));
				break;
			#endif
			#if !defined(DISABLE_RUNTIME_CPU_S8)
			case NETWORK_TYPE_S8:
				dequantize_s8(pf, (int8_t*)context->out[0], NHWC_SIZE(context->nhwc),
						LAYER_Q((*inputs)), LAYER_S((*inputs)), LAYER_Z((*inputs)));
				break;
			#endif
			#if !defined(DISABLE_RUNTIME_CPU_Q16)
			case NETWORK_TYPE_Q16:
				dequantize_q16(pf, (int16_t*)context->out[0], NHWC_SIZE(context->nhwc),
						LAYER_Q((*inputs)));
				break;
			#endif
			default:
				r = NN_E_INVALID_RUNTIME;
				break;
		}
		context->out[0] = pf;
		pf += NHWC_SIZE(context->nhwc);
		inputs++;
	}

	return r;
}

void rte_cpuq_to_cpu_float_post_execute_common(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_context_t* context;
	const layer_t* const* inputs;
	void** cl_inputs = (void**)nn->scratch.area;
	float* pf;

	if(L_OP_YOLOOUTPUT == layer->op)
	{
		return;
	}

	inputs = layer->inputs;
	while(NULL != (*inputs))
	{
		context = (layer_cpu_context_t*)(*inputs)->C->context;
		context->out[0] = *cl_inputs++;
		inputs++;
	}
}
#endif /* DISABLE_RTE_FALLBACK */

#ifndef DISABLE_DYNAMIC_SHAPE
void rte_cpu_dynamic_reshape(const layer_t* layer, layer_cpu_context_t* input_context) {
	int axis = layer_get_dynamic_axis(layer);
	if(axis > 0) {
		layer_set_dynamic_shape(layer, axis, NHWC_SIZE(input_context->nhwc));
	}
}

int rte_cpu_dynamic_conv2d(const layer_t* layer,
		layer_cpu_context_t* context, layer_cpu_context_t* input_context,
		int* padY, int* padX, int strideY, int strideX,
		int knlY, int knlX, void** O, size_t* max, size_t type_sz) {
	int r = 0;
	int axis = layer_get_dynamic_axis(layer);
	assert(axis != 3);
	if(axis > 0) {
		size_t bs;
		assert(*padY == 0xdeadbeef);
		if(0 == *padX) { /* SAME */
			context->nhwc.N = input_context->nhwc.N;
			context->nhwc.H = input_context->nhwc.H;
			context->nhwc.W = input_context->nhwc.W;
			*padY = ((context->nhwc.H-1)*strideY+knlY-input_context->nhwc.H)/2;
			*padX = ((context->nhwc.W-1)*strideX+knlX-input_context->nhwc.W)/2;
		} else { /* VALID */
			context->nhwc.N = input_context->nhwc.N;
			context->nhwc.H = (input_context->nhwc.H-knlY)/strideY + 1;
			context->nhwc.W = (input_context->nhwc.W-knlX)/strideX + 1;
			*padY = *padX = 0;
		}
		bs = NHWC_SIZE(context->nhwc);
		if(NULL == *O) {
			*O = malloc(bs*type_sz);
			*max = bs;
		} else {
			if(bs > *max) {
				free(*O);
				*O = malloc(bs*type_sz);
				*max = bs;
			}
		}
		if(NULL == *O) {
			r = NN_E_NO_MEMORY;
			context->out[0] = NULL;
		} else {
			context->out[0] = *O;
		}
	}
	return r;
}
void rte_cpu_dynamic_free(const layer_t* layer)
{
	int axis;
	layer_cpu_context_t* context = (layer_cpu_context_t*)layer->C->context;
	if(NULL != context) {
		axis = layer_get_dynamic_axis(layer);
		if(axis > 0) {
			if(NULL != context->out[0]) free(context->out[0]);
		}
	}

}
#endif /* DISABLE_DYNAMIC_SHAPE */
#endif /* DISABLE_RUNTIME_CPU */

