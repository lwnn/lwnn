/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
extern runtime_t runtime_cpu_create(const nn_t* nn);
extern int runtime_cpu_execute(const nn_t* nn);

extern runtime_t runtime_opencl_create(const nn_t* nn);
extern int runtime_opencl_execute(const nn_t* nn);
/* ============================ [ DATAS     ] ====================================================== */
static const runtime_ops_t runtime_ops[] =
{
	{	/* CPU */
		runtime_cpu_create,
		runtime_cpu_execute,
	},
	{	/* OPENCL */
		runtime_opencl_create,
		runtime_opencl_execute,
	}
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

int runtime_execute(const nn_t* nn)
{
	int r = 0;

	if(nn->runtime_type < (sizeof(runtime_ops)/sizeof(runtime_ops_t)))
	{
		r = runtime_ops[nn->runtime_type].execute(nn);
	}
	else
	{
		r = -9;
	}

	return r;
}


int runtime_execute_helper(const nn_t* nn, runtime_layer_execute_t execute)
{
	int r = 0;

	const layer_t* const* network;
	const layer_t* layer;

	network = nn->network;

	layer = *network++;
	while((NULL != layer) && (0 == r))
	{
		r = execute(nn, layer);
		layer = *network++;
	}

	return r;
}
