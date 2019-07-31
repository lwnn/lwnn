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

			NNDDO(NN_DEBUG, rte_ddo_save(nn, layer));
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
	rte_do_for_each_layer(nn, cpu_deinit_layer);
	free(nn->runtime);
}

int rte_CPU_init(const nn_t* nn)
{
	int r;
	rte_cpu_buffer_t* b;
	rte_cpu_t* rt = (rte_cpu_t*)nn->runtime;
#ifndef DISABLE_NN_LOG
	size_t sum = 0;
	size_t bufferId = -1;
#endif

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
			NNLOG(NN_DEBUG, (" buffer%d: %d\n", bufferId, b->sz));
		}
		NNLOG(NN_DEBUG, (" summary: %d\n", sum));
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

int rte_cpu_create_layer_common(const nn_t* nn, const layer_t* layer, size_t ctx_sz, size_t type_sz)
{
	int r = 0;
	layer_cpu_context_t* context;
	const char* kernel;

	r = rte_cpu_create_layer_context(nn, layer, ctx_sz, 1);

	if(0 == r)
	{
		context = (layer_cpu_context_t*)layer->C->context;

		RTE_CPU_LOG_LAYER_SHAPE(layer);

		context->out[0] = rte_cpu_create_buffer(nn, layer, NHWC_SIZE(context->nhwc)*type_sz);

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
#endif /* DISABLE_RUNTIME_CPU */

