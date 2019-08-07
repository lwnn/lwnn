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

#ifndef LCONST
#define LCONST const
#endif

#define L_INPUT(name, dtype)							\
	static layer_context_container_t l_context_##name;	\
	static LCONST int l_dims_##name[] = { name##_DIMS, 0 };	\
	static LCONST layer_t l_layer_##name = {			\
		/* name */ #name,								\
		/* inputs */ NULL,								\
		/* blobs */ l_blobs_##name,						\
		/* dims */ l_dims_##name,						\
		/* context */ &l_context_##name,				\
		/* op */ L_OP_INPUT,							\
		/* dtype */ dtype								\
	}

#define L_LAYER_SI(name, input, op)						\
	static layer_context_container_t l_context_##name;	\
	static LCONST layer_t* l_inputs_##name[] = {		\
			L_REF(input), NULL };						\
	static LCONST int l_dims_##name[] = { name##_DIMS, 0 };	\
	static LCONST layer_t l_layer_##name = {			\
		/* name */ #name,								\
		/* inputs */ l_inputs_##name,					\
		/* blobs */ l_blobs_##name,						\
		/* dims */ l_dims_##name,						\
		/* context */ &l_context_##name,				\
		/* op */ L_OP_##op,								\
		/* dtype */ L_DT_AUTO							\
	}

#define L_LAYER_MI(name, op)							\
	static layer_context_container_t l_context_##name;	\
	static LCONST int l_dims_##name[] = { name##_DIMS, 0 };	\
	static LCONST layer_t l_layer_##name = {			\
		/* name */ #name,								\
		/* inputs */ l_inputs_##name,					\
		/* blobs */ l_blobs_##name,						\
		/* dims */ l_dims_##name,						\
		/* context */ &l_context_##name,				\
		/* op */ L_OP_##op,								\
		/* dtype */ L_DT_AUTO							\
	}

#define L_OUTPUT(name, input)		L_LAYER_SI(name, input, OUTPUT)
#define L_CONV2D(name, input)		L_LAYER_SI(name, input, CONV2D)
#define L_RELU(name, input)			L_LAYER_SI(name, input, RELU)
#define L_MAXPOOL(name, input)		L_LAYER_SI(name, input, MAXPOOL)
#define L_RESHAPE(name, input)		L_LAYER_SI(name, input, RESHAPE)
#define L_DENSE(name, input)		L_LAYER_SI(name, input, DENSE)
#define L_SOFTMAX(name, input)		L_LAYER_SI(name, input, SOFTMAX)
#define L_PAD(name, input)			L_LAYER_SI(name, input, PAD)

#define L_MAXIMUM(name, inputs)							\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, MAXIMUM)

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
	NHWC_t nhwc;					\
	layer_data_type_t dtype

#define NHWC_SIZE(nhwc) (((nhwc).N)*((nhwc).H)*((nhwc).W)*((nhwc).C))

#define NHWC_BATCH_SIZE(nhwc) (((nhwc).H)*((nhwc).W)*((nhwc).C))

#define UNSUPPORTED_LAYER_OPS(runtime, op)									\
int layer_##runtime##_##op##_init(const nn_t* nn, const layer_t* layer)		\
{																			\
	NNLOG(NN_ERROR,("OP " #op " is not supported on runtime " #runtime "\n"));	\
	return NN_E_NOT_SUPPORTED;												\
}																			\
int layer_##runtime##_##op##_execute(const nn_t* nn, const layer_t* layer)	\
{																			\
	return NN_E_NOT_SUPPORTED;												\
}																			\
void layer_##runtime##_##op##_deinit(const nn_t* nn, const layer_t* layer)	\
{																			\
}
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
	LCONST struct layer** inputs;
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
int layer_get_blob_NHWC(const layer_blob_t* blob, NHWC_t* nhwc);
int layer_get_NHWC(const layer_t* layer, NHWC_t* nhwc);
size_t layer_get_size(const layer_t* layer);
#ifdef __cplusplus
}
#endif
#endif /* LAYERS_LAYER_H_ */
