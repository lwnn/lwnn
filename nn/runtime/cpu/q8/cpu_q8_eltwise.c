/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_Q8
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_q8_MAXIMUM_init(const nn_t* nn, const layer_t* layer)
{
	return -1;
}

int layer_cpu_q8_MAXIMUM_execute(const nn_t* nn, const layer_t* layer)
{
	return -1;
}

void layer_cpu_q8_MAXIMUM_deinit(const nn_t* nn, const layer_t* layer)
{

}
#endif /* DISABLE_RUNTIME_CPU_Q8 */
