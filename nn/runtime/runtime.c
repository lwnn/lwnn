/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
#define DECLARE_RUNTIME(name)									\
	extern runtime_t runtime_##name##_create(const nn_t* nn);	\
	extern int runtime_##name##_init(const nn_t* nn);			\
	extern int runtime_##name##_execute(const nn_t* nn);		\
	extern void runtime_##name##_destory(const nn_t* nn)


#define RUNTIME_REF(name)				\
	{									\
		runtime_##name##_create,		\
		runtime_##name##_init,			\
		runtime_##name##_execute,		\
		runtime_##name##_destory		\
	}
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	runtime_t (*create)(const nn_t*);
	int (*init)(const nn_t*);
	int (*execute)(const nn_t*);
	void (*destory)(const nn_t*);
} runtime_ops_t;
/* ============================ [ DECLARES  ] ====================================================== */
DECLARE_RUNTIME(cpu);
DECLARE_RUNTIME(opencl);
/* ============================ [ DATAS     ] ====================================================== */
static const runtime_ops_t runtime_ops[] =
{
#ifndef DISABLE_RUNTIME_CPU
	RUNTIME_REF(cpu),
#endif
#ifndef DISABLE_RUNTIME_OPENCL
	RUNTIME_REF(opencl),
#endif
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t runtime_create(const nn_t* nn)
{
	runtime_t runtime = NULL;

	if(nn->runtime_type < (sizeof(runtime_ops)/sizeof(runtime_ops_t)))
	{
		runtime = runtime_ops[nn->runtime_type].create(nn);
	}

	return runtime;
}

int runtime_init(const nn_t* nn)
{
	int r = NN_E_INVALID_RUNTIME;

	if(nn->runtime_type < (sizeof(runtime_ops)/sizeof(runtime_ops_t)))
	{
		r = runtime_ops[nn->runtime_type].init(nn);
	}

	return r;
}
void runtime_destory(const nn_t* nn)
{
	if(nn->runtime_type < (sizeof(runtime_ops)/sizeof(runtime_ops_t)))
	{
		runtime_ops[nn->runtime_type].destory(nn);
	}
}

int runtime_execute(const nn_t* nn)
{
	int r = NN_E_INVALID_RUNTIME;

	if(nn->runtime_type < (sizeof(runtime_ops)/sizeof(runtime_ops_t)))
	{
		r = runtime_ops[nn->runtime_type].execute(nn);
	}

	return r;
}


int runtime_do_for_each_layer(const nn_t* nn, runtime_layer_action_t action)
{
	int r = 0;

	const layer_t* const* network;
	const layer_t* layer;

	network = nn->network;

	layer = *network++;
	while((NULL != layer) && (0 == r))
	{
		r = action(nn, layer);
		layer = *network++;
	}

	return r;
}
