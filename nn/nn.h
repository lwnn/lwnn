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
#define NN_LOG(level,msg) 									\
	do {													\
		if(level >= nn_log_level) {							\
			printf msg ;									\
		} 													\
	}while(0)
#else
#define NN_LOG(level,msg)
#endif
/* ============================ [ TYPES     ] ====================================================== */
typedef struct nn {
	runtime_t runtime;
	const layer_t* const* network;

	runtime_type_t runtime_type;
} nn_t;

enum {
	NN_OK = 0,
	NN_E_INVALID_RUNTIME = -1,
	NN_E_NOT_SUPPORTED = -2,
};
/* ============================ [ DECLARES  ] ====================================================== */
extern int nn_log_level;
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
nn_t* nn_create(const layer_t* const* network, runtime_type_t runtime_type);
int nn_predict(const nn_t* nn);

void nn_set_log_level(int level);

void nn_destory(const nn_t* nn);
#ifdef __cplusplus
}
#endif
#endif /* NN_NN_H_ */
