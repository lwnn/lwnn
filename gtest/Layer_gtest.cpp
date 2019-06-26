/**
 * NNCL - Neural Network on openCL
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include <gtest/gtest.h>
#include <stdio.h>
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
#define INPUT_DIMS  1,4,4,3

#define MAX_INPUTS L_REF(input0),L_REF(input1)
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
L_INPUT (input0,  INPUT_DIMS,  L_DT_FLOAT);
L_INPUT (input1,  INPUT_DIMS,  L_DT_FLOAT);
L_ELEMENT_WISE(max, MAX_INPUTS, L_OP_MAXIMUM);
L_OUTPUT(output, max);
static const layer_t* const network1[] =
{
	L_REF(input0),
	L_REF(input1),
	L_REF(max),
	L_REF(output),
	NULL
};
/* ============================ [ FUNCTIONS ] ====================================================== */
TEST(Layer, Create)
{
	nn_set_log_level(NN_DEBUG);

	nn_t* nn = nn_create(network1);
	EXPECT_TRUE(nn != NULL);

	if(nn != NULL)
	{
		nn_predict(nn);
	}
}
