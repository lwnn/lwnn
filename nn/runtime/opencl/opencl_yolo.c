/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#if !defined(DISABLE_RUNTIME_OPENCL) && !defined(DISABLE_RTE_FALLBACK)
#include "runtime_opencl.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cl_to_cpu_float_YOLO_pre_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cl_to_cpu_float_pre_execute_common(nn, layer, 1);
}

void layer_cl_to_cpu_float_YOLO_post_execute(const nn_t* nn, const layer_t* layer)
{
	rte_cl_to_cpu_float_post_execute_common(nn, layer, 1);
}

#endif /* DISABLE_RUNTIME_CL */
