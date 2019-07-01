/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static int cpu_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	NNLOG(NN_DEBUG, (" CPU run %-16s: op=%d\n", layer->name, layer->op));
	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t rte_CPU_create(const nn_t* nn)
{
	runtime_t rt = (void*)1;

	return rt;
}

void rte_CPU_destory(const nn_t* nn)
{

}

int rte_CPU_init(const nn_t* nn)
{
	return 0;
}

int rte_CPU_execute(const nn_t* nn)
{
	return rte_do_for_each_layer(nn, cpu_execute_layer);
}
#endif /* DISABLE_RUNTIME_CPU */

