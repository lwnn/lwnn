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
extern const network_t LWNN_dense_1_q8;
extern const network_t LWNN_dense_2_q8;
extern const network_t LWNN_dense_1_float;
extern const network_t LWNN_dense_2_float;
/* ============================ [ DATAS     ] ====================================================== */
static test_case_t test_cases_maxpool[] =
{
	{
		&LWNN_dense_1_float,
		&LWNN_dense_1_q8,
		RAW_P "dense_1/golden/dense_1_input_01.raw",
		RAW_P "dense_1/golden/dense_1_output_BiasAdd_0.raw"
	},
	{
		&LWNN_dense_2_float,
		&LWNN_dense_2_q8,
		RAW_P "dense_2/golden/dense_2_input_01.raw",
		RAW_P "dense_2/golden/dense_2_output_BiasAdd_0.raw"
	},
};
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
void DenseTest(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output)
{

	if(network->layers[0]->dtype== L_DT_INT8)
	{
		nnt_siso_network_test(runtime, network, input, output, 1.0/100, 0.05);
	}
	else
	{
		nnt_siso_network_test(runtime, network, input, output);
	}
}

TEST(RuntimeCPU, DenseQ8)
{
	for(int i=0; i<ARRAY_SIZE(test_cases_maxpool); i++)
	{
		DenseTest(RUNTIME_CPU, test_cases_maxpool[i].network_q8,
				test_cases_maxpool[i].input,
				test_cases_maxpool[i].output);
	}
}

TEST(RuntimeOPENCL, Dense)
{
	for(int i=0; i<ARRAY_SIZE(test_cases_maxpool); i++)
	{
		DenseTest(RUNTIME_OPENCL, test_cases_maxpool[i].network_float,
				test_cases_maxpool[i].input,
				test_cases_maxpool[i].output);
	}
}
