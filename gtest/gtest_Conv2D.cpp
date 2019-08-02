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
extern const network_t LWNN_conv2d_1_float;
extern const network_t LWNN_conv2d_2_float;
extern const network_t LWNN_conv2d_3_float;
extern const network_t LWNN_conv2d_1_q8;
extern const network_t LWNN_conv2d_2_q8;
extern const network_t LWNN_conv2d_3_q8;
/* ============================ [ DATAS     ] ====================================================== */
static test_case_t test_cases[] =
{
	{
		&LWNN_conv2d_1_float,
		&LWNN_conv2d_1_q8,
		RAW_P "conv2d_1/golden/conv2d_1_input_01.raw",
		RAW_P "conv2d_1/golden/conv2d_1_output_BiasAdd_0.raw"
	},
	{
		&LWNN_conv2d_2_float,
		&LWNN_conv2d_2_q8,
		RAW_P "conv2d_2/golden/conv2d_2_input_01.raw",
		RAW_P "conv2d_2/golden/conv2d_2_output_BiasAdd_0.raw"
	},
	{
		&LWNN_conv2d_3_float,
		&LWNN_conv2d_3_q8,
		RAW_P "conv2d_3/golden/conv2d_3_input_01.raw",
		RAW_P "conv2d_3/golden/conv2d_3_output_BiasAdd_0.raw"
	},
};
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
void Conv2DTest(runtime_type_t runtime,
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

TEST(RuntimeCPU, Conv2DFloat)
{
	for(int i=0; i<ARRAY_SIZE(test_cases); i++)
	{
		Conv2DTest(RUNTIME_CPU, test_cases[i].network_float,
				test_cases[i].input,
				test_cases[i].output);
	}
}

TEST(RuntimeOPENCL, Conv2DFloat)
{
	for(int i=0; i<ARRAY_SIZE(test_cases); i++)
	{
		Conv2DTest(RUNTIME_OPENCL, test_cases[i].network_float,
				test_cases[i].input,
				test_cases[i].output);
	}
}

TEST(RuntimeCPU, Conv2DQ8)
{
	for(int i=0; i<ARRAY_SIZE(test_cases); i++)
	{
		Conv2DTest(RUNTIME_CPU, test_cases[i].network_q8,
				test_cases[i].input,
				test_cases[i].output);
	}
}
