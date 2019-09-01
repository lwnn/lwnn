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
#include <libgen.h>
#include <dlfcn.h>
#include <unistd.h>
#include <stdlib.h>
/* ============================ [ MACROS    ] ====================================================== */
#define EQUAL_THRESHOLD (1.0/10000)

#define RAW_P "gtest/models/"

#ifdef _WIN32
#define DLLFIX ".dll"
#define LIBFIX ""
#else
#define DLLFIX ".so"
#define LIBFIX "lib"
#endif

#define NNT_CASE_DESC(name)											\
	{																\
		"build/" RAW_P #name "/" LIBFIX #name "_q8" DLLFIX ,		\
		"build/" RAW_P #name "/" LIBFIX #name "_s8" DLLFIX ,		\
		"build/" RAW_P #name "/" LIBFIX #name "_q16" DLLFIX ,		\
		"build/" RAW_P #name "/" LIBFIX #name "_float" DLLFIX ,		\
		RAW_P #name "/golden/input.raw",							\
		RAW_P #name "/golden/output.raw",							\
		NULL														\
	}

#define NNT_CASE_DESC_ARGS(name)									\
	{																\
		"build/" RAW_P #name "/" LIBFIX #name "_q8" DLLFIX ,		\
		"build/" RAW_P #name "/" LIBFIX #name "_s8" DLLFIX ,		\
		"build/" RAW_P #name "/" LIBFIX #name "_q16" DLLFIX ,		\
		"build/" RAW_P #name "/" LIBFIX #name "_float" DLLFIX ,		\
		RAW_P #name "/golden/",										\
		RAW_P #name "/golden/",										\
		&nnt_##name##_args											\
	}

#define NNT_CASE_DEF(name)	\
		static const nnt_case_t name##_cases[]

#define NNT_TEST_DEF(runtime, name, T)					\
TEST(Runtime##runtime, name##T)							\
{														\
	for(int i=0; i<ARRAY_SIZE(name##_cases); i++)		\
	{													\
		if(g_CaseNumber != -1)							\
		{												\
			i = g_CaseNumber;							\
		}												\
		NNTTestGeneral(RUNTIME_##runtime,				\
				name##_cases[i].network##T,				\
				name##_cases[i].input,					\
				name##_cases[i].output,					\
				NNT_##name##_MAX_DIFF,					\
				NNT_##name##_MAX_QDIFF);				\
		if(g_CaseNumber != -1)							\
		{												\
			break;										\
		}												\
	}													\
}

#define NNT_MODEL_TEST_DEF(runtime, name, T)			\
TEST(Runtime##runtime, Model##name##T)					\
{														\
	for(int i=0; i<ARRAY_SIZE(name##_cases); i++)		\
	{													\
		NNTModelTestGeneral(RUNTIME_##runtime,			\
				name##_cases[i].network##T,				\
				name##_cases[i].input,					\
				name##_cases[i].output,					\
				name##_cases[i].args,					\
				NNT_##name##_TOP1,						\
				NNT_##name##_NOT_FOUND_OKAY);			\
	}													\
}

#ifndef DISABLE_RUNTIME_CPU_S8
#define NNT_TEST_CPU_S8(name)	NNT_TEST_DEF(CPU, name, S8)
#else
#define NNT_TEST_CPU_S8(name)
#endif

#ifndef DISABLE_RUNTIME_CPU_Q8
#define NNT_TEST_CPU_Q8(name)	NNT_TEST_DEF(CPU, name, Q8)
#else
#define NNT_TEST_CPU_Q8(name)
#endif

#ifndef DISABLE_RUNTIME_CPU_Q16
#define NNT_TEST_CPU_Q16(name)	NNT_TEST_DEF(CPU, name, Q16)
#else
#define NNT_TEST_CPU_Q16(name)
#endif

#ifndef DISABLE_RUNTIME_CPU_FLOAT
#define NNT_TEST_CPU_FLOAT(name)	NNT_TEST_DEF(CPU, name, Float)
#else
#define NNT_TEST_CPU_FLOAT(name)
#endif

#ifndef DISABLE_RUNTIME_OPENCL
#define NNT_TEST_OPENCL(name)	NNT_TEST_DEF(OPENCL, name, Float)
#else
#define NNT_TEST_OPENCL(name)
#endif


#define NNT_TEST_ALL(name)				\
	NNT_TEST_CPU_S8(name)				\
	NNT_TEST_CPU_Q8(name)				\
	NNT_TEST_CPU_Q16(name)				\
	NNT_TEST_CPU_FLOAT(name)			\
	NNT_TEST_OPENCL(name)				\


#ifndef DISABLE_RUNTIME_CPU_S8
#define NNT_MODEL_TEST_CPU_S8(name)	NNT_MODEL_TEST_DEF(CPU, name, S8)
#else
#define NNT_MODEL_TEST_CPU_S8(name)
#endif

#ifndef DISABLE_RUNTIME_CPU_Q8
#define NNT_MODEL_TEST_CPU_Q8(name)	NNT_MODEL_TEST_DEF(CPU, name, Q8)
#else
#define NNT_MODEL_TEST_CPU_Q8(name)
#endif

#ifndef DISABLE_RUNTIME_CPU_Q16
#define NNT_MODEL_TEST_CPU_Q16(name)	NNT_MODEL_TEST_DEF(CPU, name, Q16)
#else
#define NNT_MODEL_TEST_CPU_Q16(name)
#endif

#ifndef DISABLE_RUNTIME_CPU_FLOAT
#define NNT_MODEL_TEST_CPU_FLOAT(name)	NNT_MODEL_TEST_DEF(CPU, name, Float)
#else
#define NNT_MODEL_TEST_CPU_FLOAT(name)
#endif

#ifndef DISABLE_RUNTIME_OPENCL
#define NNT_MODEL_TEST_OPENCL(name)	NNT_MODEL_TEST_DEF(OPENCL, name, Float)
#else
#define NNT_MODEL_TEST_OPENCL(name)
#endif


#define NNT_MODEL_TEST_ALL(name)			\
	NNT_MODEL_TEST_CPU_S8(name)				\
	NNT_MODEL_TEST_CPU_Q8(name)				\
	NNT_MODEL_TEST_CPU_Q16(name)			\
	NNT_MODEL_TEST_CPU_FLOAT(name)			\
	NNT_MODEL_TEST_OPENCL(name)				\
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	const char* networkQ8;
	const char* networkS8;
	const char* networkQ16;
	const char* networkFloat;
	const char* input;
	const char* output;
	const void* args;
} nnt_case_t;
/* ============================ [ DECLARES  ] ====================================================== */
extern int g_CaseNumber; /* default -1 */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int nnt_run(const network_t* network,
			runtime_type_t runtime,
			nn_input_t** inputs,
			nn_output_t** outputs);
/* 0 means close enough, else return numbers which are not equal */
int nnt_is_equal(const float* A, const float* B, size_t sz, const float max_diff);

void nnt_fill_inputs_with_random(nn_input_t** inputs, float lo, float hi);
void* nnt_load(const char* inraw, size_t *sz);

int8_t* nnt_quantize8(float* in, size_t sz, int32_t Q, int32_t Z=0, float scale=1.0);
float* nnt_dequantize8(int8_t* in , size_t sz, int32_t Q, int32_t Z=0, float scale=1.0);
int16_t* nnt_quantize16(float* in, size_t sz, int32_t Q);
float* nnt_dequantize16(int16_t* in , size_t sz, int32_t Q);
void nnt_siso_network_test(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output,
		float max_diff = EQUAL_THRESHOLD,
		float qmax_diff = 0.15);
void NNTTestGeneral(runtime_type_t runtime,
		const char* netpath,
		const char* input,
		const char* output,
		float max_diff,
		float qmax_diff);
void NNTModelTestGeneral(runtime_type_t runtime,
		const char* netpath,
		const char* input,
		const char* output,
		float mintop1,
		float not_found_okay);
const network_t* nnt_load_network(const char* path, void** dll);
#endif /* GTEST_NN_TEST_UTIL_H_ */
