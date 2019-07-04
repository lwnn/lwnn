/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU
#include "runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct rte_cpu_buffer
{
	STAILQ_ENTRY(rte_cpu_buffer) entry;
	const layer_t* owner;
	void* data;
	size_t sz;
} rte_cpu_buffer_t;
typedef struct
{
	runtime_cpu_type_t type;
	STAILQ_HEAD(rte_cpu_buffer_head,rte_cpu_buffer) buffers;
} rte_cpu_t;
/* ============================ [ DECLARES  ] ====================================================== */
#ifndef DISABLE_RUNTIME_CPU_Q8
#define OP_DEF(op) L_OPS_DECLARE(cpu_q8_##op);
#include "opdef.h"
#undef OP_DEF
#endif
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#define OP_DEF(op) L_OPS_DECLARE(cpu_float_##op);
#include "opdef.h"
#undef OP_DEF
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
#ifndef DISABLE_RUNTIME_CPU_FLOAT
	{
		#define OP_DEF(op) L_OPS_REF(cpu_float_##op),
		#include "opdef.h"
		#undef OP_DEF
	},
#endif
};
/* ============================ [ LOCALS    ] ====================================================== */
static int cpu_get_runtime_type(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	runtime_cpu_type_t type;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	if(L_OP_INPUT == layer->op)
	{
		switch(layer->dtype)
		{
			#ifndef DISABLE_RUNTIME_CPU_Q8
			case L_DT_INT8:
				type = RET_CPU_TYPE_Q8;
				break;
			#endif
			#ifndef DISABLE_RUNTIME_CPU_FLOAT
			case L_DT_FLOAT:
				type = RTE_CPU_TYPE_FLOAT;
				break;
			#endif
			default:
				r = NN_E_NOT_SUPPORTED;
				break;
		}
		if(0 == r)
		{
			if(RTE_CPU_TYPE_UNKNOWN == rt->type)
			{
				rt->type = type;
			}
			else if(type == rt->type)
			{
				/* Pass, inputs' type are consistent */
			}
			else
			{
				r = NN_E_INPUT_TYPE_MISMATCH;
			}
		}
	}
	else
	{
		r = NN_EXIT_OK;
	}

	return r;
}

static int cpu_init_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	if(rt->type < ARRAY_SIZE(cpu_lops))
	{
		if(layer->op < ARRAY_SIZE(cpu_lops[rt->type]))
		{
			r = cpu_lops[rt->type][layer->op].init(nn, layer);
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

	if(rt->type < ARRAY_SIZE(cpu_lops))
	{
		if(layer->op < ARRAY_SIZE(cpu_lops[rt->type]))
		{
			r = cpu_lops[rt->type][layer->op].execute(nn, layer);
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

	if(rt->type < ARRAY_SIZE(cpu_lops))
	{
		if(layer->op < ARRAY_SIZE(cpu_lops[rt->type]))
		{
			cpu_lops[rt->type][layer->op].deinit(nn, layer);
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
	rte_do_for_each_layer(nn, cpu_deinit_layer);
	free(nn->runtime);
}

int rte_CPU_init(const nn_t* nn)
{
	int r;
	rte_cpu_buffer_t* b;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	rt->type = RTE_CPU_TYPE_UNKNOWN;
	STAILQ_INIT(&(rt->buffers));
	r = rte_do_for_each_layer(nn, cpu_get_runtime_type);

	if((r == NN_EXIT_OK) && (rt->type != RTE_CPU_TYPE_UNKNOWN))
	{
		r = rte_do_for_each_layer(nn, cpu_init_layer);
	}
	else
	{
		r = NN_E_INVALID_NETWORK;
	}

	if(0 == r)
	{
		STAILQ_FOREACH(b, &(rt->buffers), entry)
		{
			b->data = malloc(b->sz);
			if(b->data == NULL)
			{
				r = NN_E_NO_MEMORY;
				break;
			}
		}
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
		switch(rt->type)
		{
			#ifndef DISABLE_RUNTIME_CPU_Q8
			case RET_CPU_TYPE_Q8:
				context->dtype = L_DT_INT8;
				break;
			#endif
			#ifndef DISABLE_RUNTIME_CPU_FLOAT
			case RTE_CPU_TYPE_FLOAT:
				context->dtype = L_DT_FLOAT;
				break;
			#endif
			default:
				assert(0);
				break;
		}
		context->out = (void**)(((unsigned long long)context)+sz);
		context->nout = nout;
		memset(context->out, 0, sizeof(void*)*nout);
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

void* rte_cpu_create_buffer(const nn_t* nn, const layer_t* layer, size_t sz)
{
	int r;
	rte_cpu_buffer_t* buffer = NULL;
	rte_cpu_buffer_t* b;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;

	STAILQ_FOREACH(b, &(rt->buffers), entry)
	{
		r = rte_is_layer_consumed_from(nn, layer, b->owner);
		if(FALSE == r)
		{
			buffer = b;
			break;
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
#endif /* DISABLE_RUNTIME_CPU */

