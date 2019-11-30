/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#define DISABLE_RUNTIME_CPU_S8
#define DISABLE_RUNTIME_CPU_Q8
#define DISABLE_RUNTIME_CPU_Q16
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_BatchNorm_MAX_DIFF 5.0/100
#define NNT_BatchNorm_MAX_QDIFF 0.15
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(BatchNorm) =
{
	NNT_CASE_DESC(conv2dbn_1),
	NNT_CASE_DESC(bn_1),
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_ALL(BatchNorm)
