/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define RAW_P "gtest/models/"
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	const network_t* network_float;
	const network_t* network_q8;
	const char* input;
	const char* output;
} test_case_t;
/* ============================ [ DECLARES  ] ====================================================== */
extern const network_t LWNN_relu_1_q8;
/* ============================ [ DATAS     ] ====================================================== */
static test_case_t test_cases[] =
{
	{
		NULL,
		&LWNN_relu_1_q8,
		RAW_P "relu_1/golden/relu_1_input_01.raw",
		RAW_P "relu_1/golden/relu_1_output_Relu_0.raw"
	},
};
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
void ReLUTest(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output)
{

	if(network->layers[0]->dtype== L_DT_INT8)
	{
		nnt_siso_network_test(runtime, network, input, output, 5.0/100, 0.15);
	}
	else
	{
		nnt_siso_network_test(runtime, network, input, output);
	}
}

TEST(RuntimeCPU, ReLUQ8)
{
	for(int i=0; i<ARRAY_SIZE(test_cases); i++)
	{
		ReLUTest(RUNTIME_CPU, test_cases[i].network_q8,
				test_cases[i].input,
				test_cases[i].output);
	}
}
