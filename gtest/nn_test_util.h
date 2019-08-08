/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef GTEST_NN_TEST_UTIL_H_
#define GTEST_NN_TEST_UTIL_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#define LCONST
#include "nn.h"
#include <gtest/gtest.h>
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdio.h>
/* ============================ [ MACROS    ] ====================================================== */
#define EQUAL_THRESHOLD (1.0/10000)

#define RAW_P "gtest/models/"

#define NNT_CASE_REF(name)						\
	extern const network_t LWNN_##name##_q8;	\
	extern const network_t LWNN_##name##_q16;	\
	extern const network_t LWNN_##name##_float

#define NNT_CASE_DESC(name, output)									\
	{																\
		&LWNN_##name##_q8,											\
		&LWNN_##name##_q16,											\
		&LWNN_##name##_float,										\
		RAW_P #name "/golden/" #name "_input_01.raw",				\
		RAW_P #name "/golden/" #name "_output_" #output "_0.raw"	\
	}

#define NNT_CASE_DEF(name)	\
		static const nnt_case_t name##_cases[]

#define NNT_TEST_DEF(runtime, name, T)					\
TEST(Runtime##runtime, name##T)							\
{														\
	for(int i=0; i<ARRAY_SIZE(name##_cases); i++)		\
	{													\
		NNTTestGeneral(RUNTIME_##runtime,				\
				name##_cases[i].network##T,				\
				name##_cases[i].input,					\
				name##_cases[i].output,					\
				NNT_##name##_MAX_DIFF,					\
				NNT_##name##_MAX_QDIFF);				\
	}													\
}
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	const network_t* networkQ8;
	const network_t* networkQ16;
	const network_t* networkFloat;
	const char* input;
	const char* output;
} nnt_case_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int nnt_run(const network_t* network,
			runtime_type_t runtime,
			nn_input_t** inputs,
			nn_output_t** outputs);
nn_input_t** nnt_allocate_inputs(std::vector<const layer_t*> layers);
nn_output_t** nnt_allocate_outputs(std::vector<const layer_t*> layers);
void nnt_free_inputs(nn_input_t** inputs);
void nnt_free_outputs(nn_output_t** ouputs);
/* 0 means close enough, else return numbers which are not equal */
int nnt_is_equal(const float* A, const float* B, size_t sz, const float max_diff);

void nnt_fill_inputs_with_random(nn_input_t** inputs, float lo, float hi);
void* nnt_load(const char* inraw, size_t *sz);

int8_t* nnt_quantize8(float* in, size_t sz, int8_t Q);
float* nnt_dequantize8(int8_t* in , size_t sz, int8_t Q);
int16_t* nnt_quantize16(float* in, size_t sz, int8_t Q);
float* nnt_dequantize16(int16_t* in , size_t sz, int8_t Q);
void nnt_siso_network_test(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output,
		float max_diff = EQUAL_THRESHOLD,
		float qmax_diff = 0.15);
void NNTTestGeneral(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output,
		float max_diff,
		float qmax_diff);
#endif /* GTEST_NN_TEST_UTIL_H_ */
