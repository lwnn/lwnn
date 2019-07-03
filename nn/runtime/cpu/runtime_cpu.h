/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_RUNTIME_CPU_RUNTIME_CPU_H_
#define NN_RUNTIME_CPU_RUNTIME_CPU_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifdef __cplusplus
extern "C" {
#endif
/* ============================ [ MACROS    ] ====================================================== */
#define RTE_CPU_LOG_LAYER_SHAPE(layer) 										\
	NNLOG(NN_DEBUG, ("%s dims: [%dx%dx%dx%d]\n",							\
					layer->name,											\
					layer->C->context->nhwc.N, layer->C->context->nhwc.H,	\
					layer->C->context->nhwc.W, layer->C->context->nhwc.C))

#define LAYER_CPU_CONTEXT_MEMBER		\
		LAYER_CONTEXT_MEMBER;			\
		size_t nout;					\
		void** out
/* ============================ [ TYPES     ] ====================================================== */
typedef enum {
#ifndef DISABLE_RUNTIME_CPU_Q8
	RET_CPU_TYPE_Q8,
#endif
#ifndef DISABLE_RUNTIME_CPU_FLOAT
	RTE_CPU_TYPE_FLOAT,
#endif
	RTE_CPU_TYPE_UNKNOWN
} runtime_cpu_type_t;

typedef struct
{
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_context_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int rte_cpu_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			size_t sz, size_t nout);
void rte_cpu_destory_layer_context(const nn_t* nn, const layer_t* layer);
#ifdef __cplusplus
}
#endif
#endif /* NN_RUNTIME_CPU_RUNTIME_CPU_H_ */
