/**
 * NNCL - Neural Network on openCL
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_NN_H_
#define NN_NN_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "layer.h"
/* ============================ [ MACROS    ] ====================================================== */
#ifdef __cplusplus
extern "C" {
#endif

#define NN_DEBUG   0
#define NN_INFO    1
#define NN_WARNING 2
#define NN_ERROR   3

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
typedef struct nn nn_t;
/* ============================ [ DECLARES  ] ====================================================== */
extern int nn_log_level;
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
nn_t* nn_create(const layer_t* const* network);
int nn_predict(const nn_t* nn);

void nn_set_log_level(int level);
#ifdef __cplusplus
}
#endif
#endif /* NN_NN_H_ */
