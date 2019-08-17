/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
#define DECLARE_RUNTIME(name)									\
	extern runtime_t rte_##name##_create(const nn_t* nn);	\
	extern int rte_##name##_init(const nn_t* nn);			\
	extern int rte_##name##_execute(const nn_t* nn);		\
	extern void rte_##name##_destory(const nn_t* nn)


#define RUNTIME_REF(name)				\
	{									\
		rte_##name##_create,		\
		rte_##name##_init,			\
		rte_##name##_execute,		\
		rte_##name##_destory		\
	}
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	runtime_t (*create)(const nn_t*);
	int (*init)(const nn_t*);
	int (*execute)(const nn_t*);
	void (*destory)(const nn_t*);
} rte_ops_t;
/* ============================ [ DECLARES  ] ====================================================== */
#define RTE_DEF(rte) DECLARE_RUNTIME(rte);
	#include "rtedef.h"
#undef RTE_DEF
/* ============================ [ DATAS     ] ====================================================== */
static const rte_ops_t rte_ops[] =
{
#define RTE_DEF(rte) RUNTIME_REF(rte),
	#include "rtedef.h"
#undef RTE_DEF
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t rte_create(const nn_t* nn)
{
	runtime_t runtime = NULL;

	if(nn->runtime_type < ARRAY_SIZE(rte_ops))
	{
		runtime = rte_ops[nn->runtime_type].create(nn);
	}

	return runtime;
}

int rte_init(const nn_t* nn)
{
	int r = NN_E_INVALID_RUNTIME;

	if(nn->runtime_type < (sizeof(rte_ops)/sizeof(rte_ops_t)))
	{
		r = rte_ops[nn->runtime_type].init(nn);
	}

	return r;
}
void rte_destory(const nn_t* nn)
{
	if(nn->runtime_type < ARRAY_SIZE(rte_ops))
	{
		rte_ops[nn->runtime_type].destory(nn);
	}
}

int rte_execute(const nn_t* nn)
{
	int r = NN_E_INVALID_RUNTIME;

	if(nn->runtime_type < ARRAY_SIZE(rte_ops))
	{
		r = rte_ops[nn->runtime_type].execute(nn);
	}

	return r;
}


int rte_do_for_each_layer(const nn_t* nn, rte_layer_action_t action)
{
	int r = 0;

	const layer_t* const* layers;
	const layer_t* layer;

	layers = nn->network->layers;

	layer = *layers++;
	while((NULL != layer) && (0 == r))
	{
		r = action(nn, layer);
		layer = *layers++;
	}

	return r;
}

int rte_is_layer_consumed_from(const nn_t* nn, const layer_t* layer, const layer_t* from)
{
	int r = FALSE;

	const layer_t* const* layers;
	const layer_t** inputs;

	layers = nn->network->layers;

	while((NULL != (*layers)) && ((*layers) != from))
	{
		layers++;
	}

	do
	{
		inputs = (*layers)->inputs;
		while((*inputs != NULL) && (FALSE == r))
		{
			if(*inputs == layer)
			{
				r = TRUE;
			}

			inputs++;
		}
		layers++;
	} while((NULL != (*layers)) && (FALSE == r));

	return r;
}

#ifndef DISABLE_NN_DDO
#include <sys/stat.h>
#ifndef DISABLE_RUNTIME_CPU
#include "cpu/runtime_cpu.h"
#endif
#ifndef DISABLE_RUNTIME_OPENCL
#include "opencl/runtime_opencl.h"
#endif
static void rte_ddo_save_raw(const char* lname, int i, void* data, size_t sz)
{
	char name[128];
	FILE* fp;
	snprintf(name, sizeof(name), "tmp/%s-%d.raw", lname, i);
	#ifdef _WIN32
	mkdir("tmp");
	#else
	mkdir("tmp", S_IRWXU);
	#endif
	fp = fopen(name, "wb");

	if(fp != NULL)
	{
		fwrite(data, sz, 1, fp);
		fclose(fp);
	}
	else
	{
		printf("failed to create debug output %s\n", name);
	}
}

void rte_ddo_save(const nn_t* nn, const layer_t* layer)
{
	size_t sz = layer_get_size(layer);
	int i;

	if(L_DT_INT8 == nn->network->layers[0]->dtype)
	{
		/* pass */
	}
	else if(L_DT_INT16 == nn->network->layers[0]->dtype)
	{
		sz = sz*sizeof(int16_t);
	}
	else if(L_DT_FLOAT == nn->network->layers[0]->dtype)
	{
		sz = sz*sizeof(float);
	}
	else
	{
		assert(0);
	}

#ifndef DISABLE_RUNTIME_CPU
	if(RUNTIME_CPU == nn->runtime_type)
	{
		layer_cpu_context_t* context = (layer_cpu_context_t*)layer->C->context;
		for(i=0; i<context->nout; i++)
		{
			rte_ddo_save_raw(layer->name, i, context->out[i], sz);
		}
	}
#endif
#ifndef DISABLE_RUNTIME_OPENCL
	if(RUNTIME_OPENCL == nn->runtime_type)
	{
		int r;
		layer_cl_context_t* context;
		void* data = malloc(sz);
		if(NULL != data)
		{
			context = (layer_cl_context_t*)layer->C->context;
			for(i=0; i<context->nout; i++)
			{
				r = rte_cl_image2d_copy_out(nn, context->out[i], (float*)data, &(context->nhwc));
				if(0 == r)
				{
					rte_ddo_save_raw(layer->name, i, data, sz);
				}
				else
				{
					printf("failed to fetch CL output %s\n", layer->name);
				}
			}
			free(data);
		}
	}
#endif
}
#endif
