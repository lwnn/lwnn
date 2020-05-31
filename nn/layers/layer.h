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

#ifndef L_BLOB_NOT_BUILTIN
#define L_BLOB_DECLARE(type, name) static const type name[] = l_blob_def_##name
#else
#define L_BLOB_DECLARE(type, name) static type name[l_blob_def_##name]
#endif

#define L_LAYER_I(name, dtype, op)						\
	static layer_context_container_t l_context_##name;	\
	static LCONST int l_dims_##name[] = { name##_DIMS, 0 };	\
	static LCONST layer_t l_layer_##name = {			\
		/* name */ #name,								\
		/* inputs */ NULL,								\
		/* blobs */ l_blobs_##name,						\
		/* dims */ l_dims_##name,						\
		/* context */ &l_context_##name,				\
		/* op */ L_OP_##op,						\
		/* dtype */ dtype								\
	}

#define L_INPUT(name, dtype) L_LAYER_I(name, dtype, INPUT)
#define L_MFCC(name)  L_LAYER_I(name, L_DT_AUTO, MFCC)


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
#define L_PRELU(name, input)		L_LAYER_SI(name, input, PRELU)
#define L_MAXPOOL(name, input)		L_LAYER_SI(name, input, MAXPOOL)
#define L_AVGPOOL(name, input)		L_LAYER_SI(name, input, AVGPOOL)
#define L_RESHAPE(name, input)		L_LAYER_SI(name, input, RESHAPE)
#define L_DENSE(name, input)		L_LAYER_SI(name, input, DENSE)
#define L_SOFTMAX(name, input)		L_LAYER_SI(name, input, SOFTMAX)
#define L_PAD(name, input)			L_LAYER_SI(name, input, PAD)
#define L_DWCONV2D(name, input)		L_LAYER_SI(name, input, DWCONV2D)
#define L_UPSAMPLE(name, input)		L_LAYER_SI(name, input, UPSAMPLE)
#define L_YOLO(name, input)			L_LAYER_SI(name, input, YOLO)
#define L_DECONV2D(name, input)		L_LAYER_SI(name, input, DECONV2D)
#define L_BATCHNORM(name, input)	L_LAYER_SI(name, input, BATCHNORM)
#define L_DILCONV2D(name, input)	L_LAYER_SI(name, input, DILCONV2D)
#define L_LSTM(name, input)			L_LAYER_SI(name, input, LSTM)
#define L_TRANSPOSE(name, input)	L_LAYER_SI(name, input, TRANSPOSE)
#define L_SLICE(name, input)		L_LAYER_SI(name, input, SLICE)
#define L_CLIP(name, input)			L_LAYER_SI(name, input, CLIP)
#define L_NORMALIZE(name, input)	L_LAYER_SI(name, input, NORMALIZE)
#define L_RESIZE(name, input)		L_LAYER_SI(name, input, RESIZE)

#define L_MAXIMUM(name, inputs)							\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, MAXIMUM)

#define L_MINIMUM(name, inputs)							\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, MINIMUM)

#define L_ADD(name, inputs)								\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, ADD)

#define L_MUL(name, inputs)								\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, MUL)

#define L_CONCAT(name, inputs)							\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, CONCAT)

#define L_DETECTION(name, inputs)						\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, DETECTION)

#define L_PROPOSAL(name, inputs)						\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, PROPOSAL)

#define L_PYRAMID_ROI_ALIGN(name, inputs)				\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, PYRAMID_ROI_ALIGN)

#define L_BATCHNORM_COND(name, inputs)					\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, BATCHNORM)

#define L_CONST(name)									\
	static layer_context_container_t l_context_##name;	\
	static LCONST int l_dims_##name[] = { name##_DIMS, 0 };	\
	static LCONST layer_t l_layer_##name = {			\
		/* name */ #name,								\
		/* inputs */ NULL,								\
		/* blobs */ l_blobs_##name,						\
		/* dims */ l_dims_##name,						\
		/* context */ &l_context_##name,				\
		/* op */ L_OP_CONST,							\
		/* dtype */ L_DT_AUTO							\
	}

#define L_DETECTIONOUTPUT(name, inputs)					\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, DETECTIONOUTPUT)

#define L_YOLOOUTPUT(name, inputs)						\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, YOLOOUTPUT)

#define L_UPSAMPLE2(name, inputs)						\
	static LCONST layer_t* l_inputs_##name[] = {		\
			inputs, NULL };								\
	L_LAYER_MI(name, UPSAMPLE)

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
	layer_data_type_t dtype;		\
	size_t nout;					\
	void** out

#define NHWC_SIZE(nhwc) (((nhwc).N)*((nhwc).H)*((nhwc).W)*((nhwc).C))

#define NHWC_BATCH_SIZE(nhwc) (((nhwc).H)*((nhwc).W)*((nhwc).C))

#define NHWC_LIST(nhwc) (nhwc).N, (nhwc).H, (nhwc).W, (nhwc).C

#define L_SHAPES(layer) NHWC_LIST(layer->C->context->nhwc)

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

#ifndef DISABLE_RTE_FALLBACK
#define FALLBACK_LAYER_OPS(runtime, op, fb)									\
extern void layer_##runtime##_to_##fb##_init_common(const nn_t*,const layer_t*);			\
extern int layer_##runtime##_to_##fb##_pre_execute_common(const nn_t*,const layer_t*);		\
extern void layer_##runtime##_to_##fb##_post_execute_common(const nn_t*,const layer_t*);	\
extern void layer_##runtime##_to_##fb##_deinit_common(const nn_t*,const layer_t*);			\
extern int layer_##fb##_##op##_init(const nn_t*, const layer_t*);			\
extern int layer_##fb##_##op##_execute(const nn_t*, const layer_t*);		\
extern void layer_##fb##_##op##_deinit(const nn_t*, const layer_t*);		\
int layer_##runtime##_##op##_init(const nn_t* nn, const layer_t* layer)		\
{																			\
	int r = layer_##fb##_##op##_init(nn, layer);							\
	if(0 == r)																\
	{																		\
		layer_##runtime##_to_##fb##_init_common(nn, layer);					\
	}																		\
	return r;																\
}																			\
int layer_##runtime##_##op##_execute(const nn_t* nn, const layer_t* layer)	\
{																			\
	int r = layer_##runtime##_to_##fb##_pre_execute_common(nn, layer);		\
	if(0 == r)																\
	{																		\
		r = layer_##fb##_##op##_execute(nn, layer);							\
	}																		\
	layer_##runtime##_to_##fb##_post_execute_common(nn, layer);				\
	return r;																\
}																			\
void layer_##runtime##_##op##_deinit(const nn_t* nn, const layer_t* layer)	\
{																			\
	layer_##runtime##_to_##fb##_deinit_common(nn, layer);					\
	layer_##fb##_##op##_deinit(nn, layer);									\
}
#else
#define FALLBACK_LAYER_OPS(runtime, op, fb) UNSUPPORTED_LAYER_OPS(runtime, op)
#endif	/* DISABLE_RTE_FALLBACK */
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
	L_DT_STRING, /* for audio input */
	L_DT_AUTO
} layer_data_type_t;

typedef enum
{
	L_ACT_NONE  = 0,
	L_ACT_RELU  = 1,
	L_ACT_LEAKY = 2,
	L_ACT_SIGMOID = 3,
	L_ACT_TANH = 4,
} layer_activation_type_t;

typedef const int* layer_dimension_t;

#ifdef L_BLOB_NOT_BUILTIN
typedef int (* nn_blob_loader_t)(void* provider, void* saver, size_t size);
#endif
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

typedef void* (*layer_fetch_t)(const nn_t*, const layer_t*);

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
#ifndef DISABLE_DYNAMIC_SHAPE
int layer_get_dynamic_axis(const layer_t* layer);
void layer_set_dynamic_shape(const layer_t* layer, int axis, size_t total);
#endif
#ifdef __cplusplus
}
#endif
#endif /* LAYERS_LAYER_H_ */
