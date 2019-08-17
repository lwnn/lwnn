/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_MaxPool_MAX_DIFF  1.0/100
#define NNT_MaxPool_MAX_QDIFF 0.05
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(MaxPool) =
{
	NNT_CASE_DESC(maxpool_1),
	NNT_CASE_DESC(maxpool_2),
	NNT_CASE_DESC(maxpool1d_1),
	NNT_CASE_DESC(maxpool1d_2),
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, MaxPool, Q8)
NNT_TEST_DEF(CPU, MaxPool, Q16)
NNT_TEST_DEF(CPU, MaxPool, Float)
#ifndef DISABLE_RUNTIME_OPENCL
NNT_TEST_DEF(OPENCL, MaxPool, Float)
#endif
