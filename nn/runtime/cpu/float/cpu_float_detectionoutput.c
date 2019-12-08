/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_detection_output_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_DETECTIONOUTPUT_init(const nn_t* nn, const layer_t* layer)
{
	int r;
	#ifndef DISABLE_RTE_FALLBACK
	layer_context_t* context;
	layer_context_t* mbox_loc_context;
	layer_context_t* mbox_conf_context;
	size_t scratch_size;
	#endif

	r = rte_cpu_create_layer_context(nn, layer, sizeof(layer_cpu_float_detection_output_context_t), 0);
	#ifndef DISABLE_RTE_FALLBACK
	if(0 == r)
	{
		context = (layer_context_t*) layer->C->context;
		mbox_loc_context =
				(layer_context_t*) layer->inputs[0]->C->context;
		mbox_conf_context =
				(layer_context_t*) layer->inputs[1]->C->context;

		if((nn->runtime_type != RUNTIME_CPU) ||
			(nn->network->type != NETWORK_TYPE_FLOAT))
		{
			scratch_size = NHWC_SIZE(mbox_loc_context->nhwc) + NHWC_SIZE(mbox_conf_context->nhwc);
			scratch_size = scratch_size*sizeof(float)+2*sizeof(float*);
			nn_request_scratch(nn, scratch_size);
		}
	}
	#endif

	return r;
}

void layer_cpu_float_DETECTIONOUTPUT_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}
#ifndef DISABLE_RTE_FALLBACK
#ifndef DISABLE_RUNTIME_CPU_Q8
int layer_cpu_q8_to_cpu_float_DETECTIONOUTPUT_pre_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cpuq_to_cpu_float_pre_execute_common(nn, layer, 2);
}
void layer_cpu_q8_to_cpu_float_DETECTIONOUTPUT_post_execute(const nn_t* nn, const layer_t* layer)
{
	rte_cpuq_to_cpu_float_post_execute_common(nn, layer, 2);
}
#endif
#ifndef DISABLE_RUNTIME_CPU_S8
int layer_cpu_s8_to_cpu_float_DETECTIONOUTPUT_pre_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cpuq_to_cpu_float_pre_execute_common(nn, layer, 2);
}
void layer_cpu_s8_to_cpu_float_DETECTIONOUTPUT_post_execute(const nn_t* nn, const layer_t* layer)
{
	rte_cpuq_to_cpu_float_post_execute_common(nn, layer, 2);
}
#endif
#ifndef DISABLE_RUNTIME_CPU_Q16
int layer_cpu_q16_to_cpu_float_DETECTIONOUTPUT_pre_execute(const nn_t* nn, const layer_t* layer)
{
	return rte_cpuq_to_cpu_float_pre_execute_common(nn, layer, 2);
}
void layer_cpu_q16_to_cpu_float_DETECTIONOUTPUT_post_execute(const nn_t* nn, const layer_t* layer)
{
	rte_cpuq_to_cpu_float_post_execute_common(nn, layer, 2);
}
#endif
#endif /* DISABLE_RTE_FALLBACK */
#endif /* DISABLE_RUNTIME_CPU_FLOAT */
