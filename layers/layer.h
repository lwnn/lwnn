/**
 * NNCL - Neural Network on openCL
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
	static layer_context_t l_context_##name;			\
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
	static layer_context_t l_context_##name;			\
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

#define L_ELEMENT_WISE(name, inputs, op)				\
	static layer_context_t l_context_##name;			\
	static const struct layer* l_inputs##name[] = {		\
			inputs, NULL };								\
	static const layer_t l_layer_##name = {				\
		/* name */ #name,								\
		/* inputs */ l_inputs##name,					\
		/* blobs */ NULL,								\
		/* dims */ NULL,								\
		/* context */ &l_context_##name,				\
		/* op */ op,									\
		/* dtype */ L_DT_AUTO							\
	}

#define L_REF(name) &l_layer_##name
/* ============================ [ TYPES     ] ====================================================== */
typedef enum
{
	L_OP_INPUT,
	L_OP_MAXIMUM,
	L_OP_OUTPUT
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

typedef struct layer_context
{
	void* context;
} layer_context_t;

typedef struct layer
{
	const char* name;
	const struct layer** inputs;
	const layer_blob_t** blobs;
	const layer_dimension_t dims;
	layer_context_t* context;
	layer_operation_t op;
	layer_data_type_t dtype;
} layer_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
#ifdef __cplusplus
}
#endif
#endif /* LAYERS_LAYER_H_ */
