/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_Pad_MAX_DIFF 5.0/100
#define NNT_Pad_MAX_QDIFF 0.10
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(Pad) =
{
	NNT_CASE_DESC(pad_1),
	NNT_CASE_DESC(pad_2),
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, Pad, Q8)
NNT_TEST_DEF(CPU, Pad, Q16)
NNT_TEST_DEF(CPU, Pad, Float)
#ifndef DISABLE_RUNTIME_OPENCL
NNT_TEST_DEF(OPENCL, Pad, Float)
#endif
