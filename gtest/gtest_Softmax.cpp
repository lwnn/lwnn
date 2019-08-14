/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_Softmax_MAX_DIFF  5.0/100
#define NNT_Softmax_MAX_QDIFF 0.15
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(Softmax) =
{
	NNT_CASE_DESC(softmax_1, Softmax),
};
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, Softmax, Q8)
//NNT_TEST_DEF(CPU, Softmax, Q16)
NNT_TEST_DEF(CPU, Softmax, Float)
NNT_TEST_DEF(OPENCL, Softmax, Float)
