/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef LAYERS_LAYER_H_
#define LAYERS_LAYER_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include <stdint.h>
/* ============================ [ MACROS    ] ====================================================== */
#ifdef __cplusplus
extern "C" {
#endif

#define L_INPUT(name, shape, dtype)						\
	static layer_context_container_t l_context_##name;	\
	static const int l_dims_##name[] = { shape, 0 };	\
	static const layer_t l_layer_##name = {				\
		/* name */ #name,								\
		/* inputs */ NULL,								\
		/* blobs */ NULL,								\
		/* dims */ l_dims_##name,						\
		/* context */ &l_context_##name,				\
		/* op */ L_OP_INPUT,							\
		/* dtype */ dtype								\
	}

#define L_OUTPUT(name, input)							\
	static layer_context_container_t l_context_##name;	\
	static const struct layer* l_inputs##name[] = {		\
			L_REF(input), NULL };						\
	static const layer_t l_layer_##name = {				\
		/* name */ #name,								\
		/* inputs */ l_inputs##name,					\
		/* blobs */ NULL,								\
		/* dims */ NULL,								\
		/* context */ &l_context_##name,				\
		/* op */ L_OP_OUTPUT,							\
		/* dtype */ L_DT_AUTO							\
	}

#define L_ELEMENT_WISE(name, op)						\
	static layer_context_container_t l_context_##name;	\
	static const layer_t l_layer_##name = {				\
		/* name */ #name,								\
		/* inputs */ l_inputs##name,					\
		/* blobs */ NULL,								\
		/* dims */ NULL,								\
		/* context */ &l_context_##name,				\
		/* op */ op,									\
		/* dtype */ L_DT_AUTO							\
	}

#define L_MAXIMUM(name, inputs)							\
	static const struct layer* l_inputs##name[] = {		\
			inputs, NULL };								\
	L_ELEMENT_WISE(name, L_OP_MAXIMUM)

#define L_REF(name) &l_layer_##name

#define L_OPS_REF(name)			\
	{							\
		layer_##name##_init,	\
		layer_##name##_execute,	\
		layer_##name##_deinit	\
	}

#define L_OPS_DECLARE(name)										\
	int layer_##name##_init(const nn_t*, const layer_t*);		\
	int layer_##name##_execute(const nn_t*, const layer_t*);	\
	void layer_##name##_deinit(const nn_t*, const layer_t*)

#define LAYER_CONTEXT_MEMBER		\
	NHWC_t nhwc
/* ============================ [ TYPES     ] ====================================================== */
typedef enum
{
#define OP_DEF(op) L_OP_##op,
	#include "opdef.h"
#undef OP_DEF
	L_OP_NUMBER
} layer_operation_t;

typedef enum
{
	L_DT_INT8,
	L_DT_UINT8,
	L_DT_INT16,
	L_DT_UINT16,
	L_DT_INT32,
	L_DT_UINT32,
	L_DT_FLOAT,
	L_DT_DOUBLE,
	L_DT_AUTO
} layer_data_type_t;

typedef const int* layer_dimension_t;

typedef struct layer_blob {
	const layer_dimension_t dims;	/* 0 terminated */
	layer_data_type_t dtype;
	void* blob;
} layer_blob_t;

typedef struct
{
	int N;
	int H;
	int W;
	int C;
} NHWC_t;

typedef struct layer_context
{
	LAYER_CONTEXT_MEMBER;
} layer_context_t;

typedef struct layer_context_container
{
	layer_context_t* context;
} layer_context_container_t;

typedef struct layer
{
	const char* name;
	const struct layer** inputs;
	const layer_blob_t** blobs;
	const layer_dimension_t dims;
	layer_context_container_t* C;
	layer_operation_t op;
	layer_data_type_t dtype;
} layer_t;

typedef struct nn nn_t;

typedef struct
{
	int (*init)(const nn_t*, const layer_t*);
	int (*execute)(const nn_t*, const layer_t*);
	void (*deinit)(const nn_t*, const layer_t*);
} layer_ops_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_get_NHWC(const layer_t* layer, NHWC_t* nhwc);
size_t layer_get_size(const layer_t* layer);
#ifdef __cplusplus
}
#endif
#endif /* LAYERS_LAYER_H_ */
