/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_ALGORITHM_H_
#define NN_ALGORITHM_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
/* adjust pointer address by offset */
#define APABO(addr, offset) ((void*)(((size_t)(addr))+(offset)))
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int alg_concat(const nn_t* nn, const layer_t* layer, int axis,
		void* pout, void* (*fetch_input)(const nn_t* nn, const layer_t* layer),
		size_t type_size);
int alg_up_sampling(void* pout, void* pin, NHWC_t *outNHWC, NHWC_t *inNHWC, size_t type_size);
#endif /* NN_ALGORITHM_H_ */
