/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#define DISABLE_RUNTIME_CPU_S8
#define DISABLE_RUNTIME_CPU_Q8
#define DISABLE_RUNTIME_CPU_Q16
#define DISABLE_RUNTIME_OPENCL
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_DilatedConv2D_MAX_DIFF 5.0/100
#define NNT_DilatedConv2D_MAX_QDIFF 0.15
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(DilatedConv2D) =
{
	NNT_CASE_DESC(dilconv2d_1),
	NNT_CASE_DESC(dilconv2d_2),
	NNT_CASE_DESC(dilconv2d_3),
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_ALL(DilatedConv2D)
