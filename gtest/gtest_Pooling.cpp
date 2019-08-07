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
NNT_CASE_REF(maxpool_1);
NNT_CASE_REF(maxpool_2);
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(MaxPool) =
{
	NNT_CASE_DESC(maxpool_1, MaxPool),
	NNT_CASE_DESC(maxpool_2, MaxPool),
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, MaxPool, Q8)
NNT_TEST_DEF(CPU, MaxPool, Float)
NNT_TEST_DEF(OPENCL, MaxPool, Float)
