/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define INPUT_DIMS 2,4,5,7
#define input0_DIMS INPUT_DIMS
#define input1_DIMS INPUT_DIMS

#define MAX_INPUTS L_REF(input0),L_REF(input1)

#define MAX_VERIFY_OUTPUTS(type, inputs, outputs)			\
do {													\
	size_t sz = layer_get_size(inputs[0]->layer);		\
														\
	int nequal = 0;										\
	type* A = (type*)inputs[0]->data;					\
	type* B = (type*)inputs[1]->data;					\
	type* O = (type*)outputs[0]->data;					\
	for(int i=0; i<sz; i++)								\
	{													\
		type a = A[i];									\
		type b = B[i];									\
		type o = O[i];									\
		type max = (type)std::fmax((float)a,(float)b);	\
														\
		if(std::fabs(max-o) > EQUAL_THRESHOLD)			\
		{												\
			nequal++;									\
			printf("@%d %f != %f=fmax(%f, %f)\n",		\
					i, (float)max, (float)o, (float)a, (float)b);	\
		}															\
	}																\
	EXPECT_TRUE(0 == nequal);										\
} while(0)
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
L_INPUT (input0, L_DT_FLOAT);
L_INPUT (input1, L_DT_FLOAT);
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

static const int test_dims[][4] =
{
	{INPUT_DIMS},
	{46,57,58,93},
	{23,45,78,0},
	{34,98,0, 0}
};

/* ============================ [ FUNCTIONS ] ====================================================== */
void EltmentWiseTest(runtime_type_t runtime, const int dims[4])
{
	nn_set_log_level(NN_DEBUG);

	memcpy(l_dims_input0, dims, sizeof(int)*4);
	memcpy(l_dims_input1, dims, sizeof(int)*4);

	nn_input_t** inputs = nnt_allocate_inputs({L_REF(input0), L_REF(input1)});
	nn_output_t** outputs = nnt_allocate_outputs({L_REF(output)});

	nnt_fill_inputs_with_random(inputs, -100, 100);
	int r = nnt_run(network1, runtime, inputs, outputs);

	if(0 == r)
	{
		if(l_layer_input0.dtype == L_DT_FLOAT)
		{
			MAX_VERIFY_OUTPUTS(float, inputs, outputs);
		}
		else if(l_layer_input0.dtype == L_DT_INT8)
		{
			MAX_VERIFY_OUTPUTS(int8_t, inputs, outputs);
		}
		else
		{
			assert(0);
		}

	}

	nnt_free_inputs(inputs);
	nnt_free_outputs(outputs);

}

TEST(RuntimeOPENCL, ElementWiseMax)
{
	for(int i=0; i<ARRAY_SIZE(test_dims); i++)
	{
		EltmentWiseTest(RUNTIME_OPENCL, test_dims[i]);
	}
}

TEST(RuntimeCPU, ElementWiseMax)
{
	for(int i=0; i<ARRAY_SIZE(test_dims); i++)
	{
		EltmentWiseTest(RUNTIME_CPU, test_dims[i]);
	}
}

TEST(RuntimeCPUQ8, ElementWiseMax)
{
	l_layer_input0.dtype = L_DT_INT8;
	l_layer_input1.dtype = L_DT_INT8;

	for(int i=0; i<ARRAY_SIZE(test_dims); i++)
	{
		EltmentWiseTest(RUNTIME_CPU, test_dims[i]);
	}

	l_layer_input0.dtype = L_DT_FLOAT;
	l_layer_input1.dtype = L_DT_FLOAT;

}
