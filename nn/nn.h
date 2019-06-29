/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_NN_H_
#define NN_NN_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "layer.h"
#include "runtime.h"
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdarg.h>
#include <assert.h>
/* ============================ [ MACROS    ] ====================================================== */
#ifdef __cplusplus
extern "C" {
#endif

#define NN_DEBUG			0
#define NN_INFO				100
#define NN_WARNING			200
#define NN_CRITICAL			300
#define NN_ERROR			400

#ifndef DISABLE_NN_LOG
#define NNLOG(level,msg) 									\
	do {													\
		if(level >= nn_log_level) {							\
			printf msg ;									\
		} 													\
	}while(0)
#else
#define NN_LOG(level,msg)
#endif

#ifndef ARRAY_SIZE
#define ARRAY_SIZE(a) (sizeof(a)/sizeof(a[0]))
#endif
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	const layer_t* layer;
	void* data;
} nn_input_t;

typedef struct nn {
	runtime_t runtime;
	const layer_t* const* network;

	runtime_type_t runtime_type;

	const nn_input_t* const* inputs;
} nn_t;

enum {
	NN_OK = 0,
	NN_E_INVALID_RUNTIME = -1,
	NN_E_NOT_SUPPORTED = -2,
	NN_E_NO_MEMORY = -3,
	NN_E_INVALID_DIMENSION = -4,
	NN_E_INVALID_LAYER = -5,
	NN_E_CREATE_CL_CONTEXT_FAILED = -6,
	NN_E_CL_SET_ARGS_FAILED = -7,
	NN_E_CL_EXECUTE_FAILED = -8,
};
/* ============================ [ DECLARES  ] ====================================================== */
extern int nn_log_level;
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
nn_t* nn_create(const layer_t* const* network, runtime_type_t runtime_type);
int nn_predict(nn_t* nn, const nn_input_t* const * inputs);

void nn_set_log_level(int level);

void nn_destory(nn_t* nn);

void* nn_allocate_input(const layer_t* layer);
void nn_free_input(void* input);
void* nn_get_input_data(const nn_t* nn, const layer_t* layer);

#ifdef __cplusplus
}
#endif
#endif /* NN_NN_H_ */
