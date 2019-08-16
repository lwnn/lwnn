/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_Concat_MAX_DIFF 5.0/100
#define NNT_Concat_MAX_QDIFF 0.15
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(Concat) =
{
	NNT_CASE_DESC(concat_1, concat),
	NNT_CASE_DESC(concat_2, concat),
	NNT_CASE_DESC(concat_3, concat),
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, Concat, Q8)
NNT_TEST_DEF(CPU, Concat, Q16)
NNT_TEST_DEF(CPU, Concat, Float)
NNT_TEST_DEF(OPENCL, Concat, Float)
