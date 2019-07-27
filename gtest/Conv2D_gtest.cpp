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
		RAW_P "conv2d_1/golden/input_1_01.raw",
		RAW_P "conv2d_1/golden/conv2d_1_BiasAdd_0.raw"
	},
	{
		&LWNN_conv2d_2_float,
		&LWNN_conv2d_2_q8,
		RAW_P "conv2d_2/golden/input_2_01.raw",
		RAW_P "conv2d_2/golden/conv2d_2_BiasAdd_0.raw"
	},
	{
		&LWNN_conv2d_3_float,
		&LWNN_conv2d_3_q8,
		RAW_P "conv2d_3/golden/input_3_01.raw",
		RAW_P "conv2d_3/golden/conv2d_3_BiasAdd_0.raw"
	},
};
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
void Conv2DTest(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output)
{
	nn_set_log_level(NN_DEBUG);

	nn_input_t** inputs = nnt_allocate_inputs({network->layers[0]});
	nn_output_t** outputs = nnt_allocate_outputs({network->layers[2]});

	size_t sz_in;
	float* IN = (float*)nnt_load(input, &sz_in);
	ASSERT_EQ(sz_in, layer_get_size((inputs[0])->layer)*sizeof(float));

	int8_t* in8 = NULL;
	if(network->layers[0]->dtype== L_DT_INT8)
	{
		int8_t* blob = (int8_t*)network->layers[0]->blobs[0]->blob;
		in8 = nnt_quantize8(IN, sz_in/sizeof(float), blob[0]);
		memcpy(inputs[0]->data, in8, sz_in/sizeof(float));
	}
	else
	{
		memcpy(inputs[0]->data, IN, sz_in);
	}

	int r = nnt_run(network, runtime, inputs, outputs);

	if(0 == r)
	{
		size_t sz_out;
		float* OUT = (float*)nnt_load(output, &sz_out);
		ASSERT_EQ(sz_out, layer_get_size((outputs[0])->layer)*sizeof(float));

		if(in8 != NULL)
		{
			int8_t* blob = (int8_t*)outputs[0]->layer->blobs[0]->blob;
			float* out = nnt_dequantize8((int8_t*)outputs[0]->data, layer_get_size(outputs[0]->layer), blob[0]);
			r = nnt_is_equal(OUT, out,
					layer_get_size(outputs[0]->layer), 5/100.0, 1);
			free(out);
			/* if 85% data is okay, pass test */
			EXPECT_LE(r, layer_get_size(outputs[0]->layer)*15/100);
		}
		else
		{
			r = nnt_is_equal(OUT, (float*)outputs[0]->data,
					layer_get_size(outputs[0]->layer), EQUAL_THRESHOLD);
			EXPECT_EQ(0, r);
		}

		free(OUT);
	}

	if(in8 != NULL)
	{
		free(in8);
	}

	free(IN);

	nnt_free_inputs(inputs);
	nnt_free_outputs(outputs);
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

TEST(RuntimeCPU, Conv2DQ8)
{
	for(int i=0; i<ARRAY_SIZE(test_cases); i++)
	{
		Conv2DTest(RUNTIME_CPU, test_cases[i].network_q8,
				test_cases[i].input,
				test_cases[i].output);
	}
}
