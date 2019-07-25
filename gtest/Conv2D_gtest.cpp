/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define RAW_P "gtest/models/"
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
extern const layer_t* const LWNN_conv2d_1[];
extern const layer_t* const LWNN_conv2d_2[];
/* ============================ [ DATAS     ] ====================================================== */
static struct {
	const layer_t* const* network;
	const char* input;
	const char* output;
} test_cases[] =
{
	{
		LWNN_conv2d_1,
		RAW_P "conv2d_1/input_1_01.raw",
		RAW_P "conv2d_1/conv2d_1_BiasAdd_0.raw"
	},
	{
		LWNN_conv2d_2,
		RAW_P "conv2d_2/input_2_01.raw",
		RAW_P "conv2d_2/conv2d_2_BiasAdd_0.raw"
	},
};
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
void Conv2DTest(runtime_type_t runtime,
		const layer_t* const* network,
		const char* input,
		const char* output)
{
	nn_set_log_level(NN_DEBUG);

	nn_input_t** inputs = nnt_allocate_inputs({network[0]});
	nn_output_t** outputs = nnt_allocate_outputs({network[2]});

	size_t sz_in;
	float* IN = (float*)nnt_load(input, &sz_in);
	ASSERT_EQ(sz_in, layer_get_size((inputs[0])->layer)*sizeof(float));
	memcpy(inputs[0]->data, IN, sz_in);

	int r = nnt_run(network, runtime, inputs, outputs);

	if(0 == r)
	{
		size_t sz_out;
		float* OUT = (float*)nnt_load(output, &sz_out);
		ASSERT_EQ(sz_out, layer_get_size((outputs[0])->layer)*sizeof(float));

		r = nnt_is_equal(OUT, (float*)outputs[0]->data,
					layer_get_size(outputs[0]->layer), 0.000001);
		EXPECT_TRUE(0 == r);
	}

	nnt_free_inputs(inputs);
	nnt_free_outputs(outputs);
}

TEST(RuntimeCPU, Conv2D)
{
	for(int i=0; i<ARRAY_SIZE(test_cases); i++)
	{
		Conv2DTest(RUNTIME_CPU, test_cases[i].network,
				test_cases[i].input,
				test_cases[i].output);
	}
}
