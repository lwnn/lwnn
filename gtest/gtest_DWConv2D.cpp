/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_DWConv2D_MAX_DIFF 5.0/100
#define NNT_DWConv2D_MAX_QDIFF 0.15
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
NNT_CASE_REF(dwconv2d_1);
NNT_CASE_REF(dwconv2d_2);
NNT_CASE_REF(dwconv2d_3);
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(DWConv2D) =
{
	NNT_CASE_DESC(dwconv2d_1, BiasAdd),
	NNT_CASE_DESC(dwconv2d_2, BiasAdd),
	NNT_CASE_DESC(dwconv2d_3, BiasAdd)
};
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
NNT_TEST_DEF(CPU, DWConv2D, Q8)
NNT_TEST_DEF(CPU, DWConv2D, Q16)
NNT_TEST_DEF(CPU, DWConv2D, Float)
NNT_TEST_DEF(OPENCL, DWConv2D, Float)
