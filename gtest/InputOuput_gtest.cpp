/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include <gtest/gtest.h>
#include <stdio.h>
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
#define INPUT_DIMS 2,4,5,7

/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
L_INPUT (input0,  INPUT_DIMS,  L_DT_FLOAT);
L_OUTPUT(output, input0);
static const layer_t* const network1[] =
{
	L_REF(input0),
	L_REF(output),
	NULL
};

/* ============================ [ FUNCTIONS ] ====================================================== */
void InputOutputTest1(runtime_type_t runtime)
{
	nn_set_log_level(NN_DEBUG);

	float* data0 = (float*)nn_allocate_input(L_REF(input0));
	ASSERT_TRUE(data0 != NULL);

	for(int i=0; i<layer_get_size(L_REF(input0)); i++)
	{
		data0[i] = i+1;
	}

	nn_input_t input0 = { L_REF(input0), data0 };
	nn_input_t* inputs[] = { &input0, NULL };

	nn_t* nn = nn_create(network1, runtime);
	EXPECT_TRUE(nn != NULL);

	if(nn != NULL)
	{
		nn_predict(nn, inputs);
		nn_destory(nn);
	}

	nn_free_input(data0);
}

TEST(RuntimeOPENCL, InputOutput)
{
	InputOutputTest1(RUNTIME_OPENCL);
}

TEST(RuntimeCPU, InputOutput)
{
	InputOutputTest1(RUNTIME_CPU);
}
