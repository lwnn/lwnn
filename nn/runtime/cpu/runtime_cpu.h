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

#define LAYER_CPU_Q8_CONTEXT_MEMBER		\
		LAYER_CPU_CONTEXT_MEMBER;		\
		int8_t Q

#ifndef FLT_MAX
#define FLT_MAX  3.40282347e+38F
#endif
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

typedef struct
{
	LAYER_CPU_Q8_CONTEXT_MEMBER;
} layer_cpu_q8_context_t;

typedef struct rte_cpu_buffer
{
	STAILQ_ENTRY(rte_cpu_buffer) entry;
	const layer_t* owner;
	void* data;
	size_t sz;
} rte_cpu_buffer_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int rte_cpu_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			size_t sz, size_t nout);
void rte_cpu_destory_layer_context(const nn_t* nn, const layer_t* layer);
void* rte_cpu_create_buffer(const nn_t* nn, const layer_t* layer, size_t sz);
void rte_cpu_take_buffer(rte_cpu_buffer_t* buffer, const layer_t* layer);
void rte_cpu_release_buffer(rte_cpu_buffer_t* buffer);

int rte_cpu_create_layer_common(const nn_t* nn, const layer_t* layer, size_t ctx_sz, size_t type_sz);
#ifdef __cplusplus
}
#endif
#endif /* NN_RUNTIME_CPU_RUNTIME_CPU_H_ */
