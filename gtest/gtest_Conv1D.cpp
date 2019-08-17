/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_Conv1D_MAX_DIFF 5.0/100
#define NNT_Conv1D_MAX_QDIFF 0.15
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(Conv1D) =
{
	NNT_CASE_DESC(conv1d_1),
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, Conv1D, Q8)
NNT_TEST_DEF(CPU, Conv1D, Q16)
NNT_TEST_DEF(CPU, Conv1D, Float)
#ifndef DISABLE_RUNTIME_OPENCL
NNT_TEST_DEF(OPENCL, Conv1D, Float)
#endif
