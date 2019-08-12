/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_BatchNorm_MAX_DIFF 5.0/100
#define NNT_BatchNorm_MAX_QDIFF 0.15
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
NNT_CASE_REF(conv2dbn_1);
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(BatchNorm) =
{
	NNT_CASE_DESC(conv2dbn_1, cond_Merge),
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, BatchNorm, Q8)
NNT_TEST_DEF(CPU, BatchNorm, Q16)
NNT_TEST_DEF(CPU, BatchNorm, Float)
NNT_TEST_DEF(OPENCL, BatchNorm, Float)
