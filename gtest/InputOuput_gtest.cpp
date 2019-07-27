/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define INPUT_DIMS 2,4,5,7
#define input0_DIMS INPUT_DIMS
#define output_DIMS INPUT_DIMS

#define l_blobs_input0 NULL
#define l_blobs_output NULL
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
L_INPUT (input0, L_DT_FLOAT);
L_OUTPUT(output, input0);
static const layer_t* const network1_layers[] =
{
	L_REF(input0),
	L_REF(output),
	NULL
};

static const network_t network1 =
{
	"input_output",
	network1_layers
};

static const int test_dims[][4] =
{
	{INPUT_DIMS},
	{46,57,58,93},
	{23,45,78,0},
	{34,98,0, 0}
};

/* ============================ [ FUNCTIONS ] ====================================================== */
static void InputOutputTest(runtime_type_t runtime, const int dims[4])
{
	memcpy(l_dims_input0, dims, sizeof(int)*4);
	memcpy(l_dims_output, dims, sizeof(int)*4);

	nn_input_t** inputs = nnt_allocate_inputs({L_REF(input0)});
	nn_output_t** outputs = nnt_allocate_outputs({L_REF(output)});

	nnt_fill_inputs_with_random(inputs, -10, 10);
	int r = nnt_run(&network1, runtime, inputs, outputs);

	if(0 == r)
	{
		r = nnt_is_equal((float*)inputs[0]->data, (float*)outputs[0]->data,
					layer_get_size(inputs[0]->layer), EQUAL_THRESHOLD);
		EXPECT_TRUE(0 == r);
	}

	nnt_free_inputs(inputs);
	nnt_free_outputs(outputs);
}

TEST(RuntimeOPENCL, InputOutput)
{
	for(int i=0; i<ARRAY_SIZE(test_dims); i++)
	{
		InputOutputTest(RUNTIME_OPENCL, test_dims[i]);
	}
}

TEST(RuntimeCPU, InputOutput)
{
	for(int i=0; i<ARRAY_SIZE(test_dims); i++)
	{
		InputOutputTest(RUNTIME_CPU, test_dims[i]);
	}
}
