/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
#include "algorithm.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct alg_transpose_case {
	const char* input;
	const char* output0; /* for from nhwc to nchw */
	NHWC_t nhwc;
} alg_transpose_case_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
static const alg_transpose_case_t alg_transpose_cases[] =
{
	{
		RAW_P "transpose/golden/input0.raw",
		RAW_P "transpose/golden/output0_0.raw",
		{1,2,4,3},
	},
};
/* ============================ [ LOCALS    ] ====================================================== */
void TestTransposeFromNHWC2NCHW(const char* input, const char* output, const NHWC_t* nhwc)
{
	int r;
	size_t sz_in, sz_out;
	float* IN = (float*)nnt_load(input, &sz_in);
	float* G = (float*)nnt_load(output, &sz_out);
	ASSERT_EQ(sz_in, sz_out);
	ASSERT_EQ(sz_in, NHWC_SIZE(*nhwc)*sizeof(float));

	float* OUT = (float*) malloc(sz_out);
	ASSERT_NE(OUT, nullptr);

	r = alg_transpose(OUT, IN, nhwc, sizeof(float), ALG_TRANSPOSE_FROM_NHWC_TO_NCHW);
	EXPECT_EQ(0, r);
	r = nnt_is_equal(OUT, G,
			sz_out/sizeof(float), 1.0/1000);
	EXPECT_EQ(0, r);

	free(OUT);
	free(IN);
	free(G);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
TEST(Algorighm, Transpose)
{
	for(int i=0; i<ARRAY_SIZE(alg_transpose_cases); i++)
	{
		TestTransposeFromNHWC2NCHW(
				alg_transpose_cases[i].input,
				alg_transpose_cases[i].output0,
				&alg_transpose_cases[i].nhwc);
	}
}
