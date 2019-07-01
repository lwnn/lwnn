/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
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

	nnt_fill_inputs_with_random_f(inputs);
	int r = nnt_run(network1, runtime, inputs, outputs);

	if(0 == r)
	{

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
