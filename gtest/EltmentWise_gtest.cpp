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

#define MAX_INPUTS L_REF(input0),L_REF(input1)
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
L_INPUT (input0,  INPUT_DIMS,  L_DT_FLOAT);
L_INPUT (input1,  INPUT_DIMS,  L_DT_FLOAT);
L_MAXIMUM(max, MAX_INPUTS);
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
void EltmentWiseTest1(runtime_type_t runtime)
{
	//nn_set_log_level(NN_DEBUG);

	float* data0 = (float*)nn_allocate_input(L_REF(input0));
	ASSERT_TRUE(data0 != NULL);
	float* data1 = (float*)nn_allocate_input(L_REF(input0));
	ASSERT_TRUE(data1 != NULL);

	for(int i=0; i<layer_get_size(L_REF(input0)); i++)
	{
		data0[i] = i+1;
		data1[i] = i+1;
	}

	nn_input_t input0 = { L_REF(input0), data0 };
	nn_input_t input1 = { L_REF(input1), data1 };

	nn_input_t* inputs[] = { &input0, &input1, NULL };

	nn_t* nn = nn_create(network1, runtime);
	EXPECT_TRUE(nn != NULL);

	if(nn != NULL)
	{
		float* out = (float*) nn_allocate_output(L_REF(output));
		ASSERT_TRUE(out != NULL);
		nn_output_t out0 = { L_REF(output), out };
		nn_output_t* outputs[] = { &out0, NULL };

		int r = nn_predict(nn, inputs, outputs);
		EXPECT_TRUE(0 == r);
		nn_destory(nn);

		nn_free_output(out);
	}

	nn_free_input(data0);
	nn_free_input(data1);

}

TEST(RuntimeOPENCL, ElementWiseMax)
{
	EltmentWiseTest1(RUNTIME_OPENCL);
}

TEST(RuntimeCPU, ElementWiseMax)
{
	EltmentWiseTest1(RUNTIME_CPU);
}
