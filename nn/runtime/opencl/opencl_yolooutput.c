/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_to_cpu_float_YOLOOUTPUT_pre_execute(const nn_t* nn, const layer_t* layer)
{
	(void)nn; (void)layer;
	return 0;
}

void layer_cl_to_cpu_float_YOLOOUTPUT_post_execute(const nn_t* nn, const layer_t* layer)
{
	(void)nn; (void)layer;
	/* pass */
}

#endif /* DISABLE_RUNTIME_OPENCL */
