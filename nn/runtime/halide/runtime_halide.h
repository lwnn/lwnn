/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_RUNTIME_HALIDE_RUNTIME_HALIDE_H_
#define NN_RUNTIME_HALIDE_RUNTIME_HALIDE_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include <Halide.h>
#include <string>
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
#define LAYER_HALIDE_CONTEXT_MEMBER		\
		LAYER_CONTEXT_MEMBER

#define RAW_NAME(nn, layer, i) \
	(std::string("tmp/")+nn->network->name+"-"+layer->name+"-"+i+".raw").c_str()
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	LAYER_HALIDE_CONTEXT_MEMBER;
} layer_halide_context_t;

typedef struct rte_halide_buffer
{
	STAILQ_ENTRY(rte_cl_image) entry;
	const layer_t* owner;
	int H;
	int W;
} rte_halide_buffer_t;

/* ============================ [ DECLARES  ] ====================================================== */
extern "C" void rte_ddo_save(const nn_t* nn, const layer_t* layer);
extern "C" void rte_save_raw(const char* name, void* data, size_t sz);
extern "C" void rte_ddo_save_raw(const nn_t* nn, const layer_t* layer, int i, void* data, size_t sz);
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int rte_halide_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			size_t sz, size_t nout);
void rte_halide_destory_layer_context(const nn_t* nn, const layer_t* layer);
int rte_halide_create_layer_common(const nn_t* nn, const layer_t* layer, size_t ctx_sz);
Halide::Buffer<float>* rte_halide_create_buffer_from_blob(const nn_t* nn, const layer_blob_t* blob);
#endif /* NN_RUNTIME_HALIDE_RUNTIME_HALIDE_H_ */
