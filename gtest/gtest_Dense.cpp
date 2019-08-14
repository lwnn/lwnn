/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_Dense_MAX_DIFF  5.0/100
#define NNT_Dense_MAX_QDIFF 0.05
/* ============================ [ TYPES     ] ====================================================== */

/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(Dense) =
{
	NNT_CASE_DESC(dense_1, BiasAdd),
	NNT_CASE_DESC(dense_2, BiasAdd),
};

/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, Dense, Q8)
NNT_TEST_DEF(CPU, Dense, Q16)
NNT_TEST_DEF(CPU, Dense, Float)
NNT_TEST_DEF(OPENCL, Dense, Float)
