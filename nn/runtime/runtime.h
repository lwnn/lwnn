/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_RUNTIME_H_
#define NN_RUNTIME_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
#ifdef __cplusplus
extern "C" {
#endif
/* ============================ [ TYPES     ] ====================================================== */
typedef enum {
	RUNTIME_CPU,
	RUNTIME_OPENCL
} runtime_type_t;

typedef void* runtime_t;

typedef struct nn nn_t;

typedef struct
{
	runtime_t (*create)(const nn_t*);
	int (*execute)(const nn_t*);
} runtime_ops_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t runtime_create(const nn_t* nn);
int runtime_execute(const nn_t* nn);
#ifdef __cplusplus
}
#endif
#endif /* NN_RUNTIME_H_ */
