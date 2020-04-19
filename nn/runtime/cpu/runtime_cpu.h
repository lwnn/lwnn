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
#define LAYER_CPU_CONTEXT_MEMBER		\
		LAYER_CONTEXT_MEMBER

#define LAYER_CPU_Q_CONTEXT_MEMBER		\
		LAYER_CPU_CONTEXT_MEMBER

#define LAYER_CPU_Q8_CONTEXT_MEMBER LAYER_CPU_Q_CONTEXT_MEMBER
#define LAYER_CPU_Q16_CONTEXT_MEMBER LAYER_CPU_Q_CONTEXT_MEMBER

#define LAYER_CPU_S8_CONTEXT_MEMBER		\
		LAYER_CPU_Q_CONTEXT_MEMBER

#ifndef FLT_MAX
#define FLT_MAX  3.40282347e+38F
#endif

#ifndef DISABLE_RTE_FALLBACK
#define IS_LAYER_WITH_REAL_BUFFER(layer) \
	( (L_OP_MFCC == layer->op) || \
	  (L_OP_TRANSPOSE == layer->op) || \
	  (L_OP_LSTM == layer->op) )

#define IS_LAYER_WITH_INPUT_BUFFER_READY(layer) \
	( (L_OP_YOLOOUTPUT == layer->op) || \
	  (L_OP_MFCC == layer->op) )

#define IS_LAYER_WITHOUT_BUFFER(layer) \
	( (L_OP_YOLOOUTPUT == layer->op) || \
	  (L_OP_DETECTIONOUTPUT == layer->op) )
#endif /* DISABLE_RTE_FALLBACK */

#ifndef DISABLE_DYNAMIC_SHAPE
#define LAYER_CPU_DYNMIC_SHAPE_COMMON_MEMBER size_t allocated
#else
#define LAYER_CPU_DYNMIC_SHAPE_COMMON_MEMBER
#endif
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_context_t;

typedef struct
{
	LAYER_CPU_Q_CONTEXT_MEMBER;
} layer_cpu_q_context_t;

typedef layer_cpu_q_context_t layer_cpu_q8_context_t;
typedef layer_cpu_q_context_t layer_cpu_q16_context_t;

typedef struct
{
	LAYER_CPU_S8_CONTEXT_MEMBER;
} layer_cpu_s8_context_t;

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
void rte_cpu_take_buffer(rte_cpu_buffer_t* buffer, const layer_t* layer, int id);
void rte_cpu_release_buffer(rte_cpu_buffer_t* buffer);

int rte_cpu_create_layer_common(const nn_t* nn, const layer_t* layer, size_t ctx_sz, size_t type_sz);
void* rte_cpu_fetch_out0(const nn_t* nn, const layer_t* layer);

#ifndef DISABLE_DYNAMIC_SHAPE
void rte_cpu_dynamic_reshape(const layer_t* layer, layer_cpu_context_t* input_context);
void rte_cpu_dynamic_shape_copy(const layer_t* layer, layer_cpu_context_t* input_context);
void rte_cpu_dynamic_batch(const layer_t* layer, layer_cpu_context_t* input_context);
int rte_cpu_dynamic_memory(void** mem, size_t required, size_t* allocated, size_t type_sz);
int rte_cpu_dynamic_conv2d_or_pool(const layer_t* layer,
		layer_cpu_context_t* context, layer_cpu_context_t* input_context,
		int* padY, int* padX, int strideY, int strideX,
		int knlY, int knlX);
void rte_cpu_dynamic_free(const layer_t* layer);
#else
#define rte_cpu_dynamic_reshape(layer, input_context)
#define rte_cpu_dynamic_shape_copy(layer, input_context)
#define rte_cpu_dynamic_batch(layer, input_context)
#define rte_cpu_dynamic_memory(mem, required, allocated, type_sz) 0
#define rte_cpu_dynamic_conv2d(layer, context, input_context, \
	padY, padX, strideY, strideX, knlY, knlX, O, max, type_sz) 0
#define rte_cpu_dynamic_free(layer)
#endif

#ifndef DISABLE_RTE_FALLBACK
void rte_cpuq_to_cpu_float_init_common(const nn_t* nn, const layer_t* layer);
int rte_cpuq_to_cpu_float_pre_execute_common(const nn_t* nn, const layer_t* layer);
void rte_cpuq_to_cpu_float_post_execute_common(const nn_t* nn, const layer_t* layer);
void rte_cpuq_to_cpu_float_deinit_common(const nn_t* nn, const layer_t* layer);
#endif
#ifdef __cplusplus
}
#endif
#endif /* NN_RUNTIME_CPU_RUNTIME_CPU_H_ */
