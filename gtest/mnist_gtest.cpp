/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define RAW_P "gtest/models/mnist/golden/"
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
extern const network_t LWNN_mnist_q8;
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
void MNISTTest(runtime_type_t runtime,
		const network_t* network)
{
	int r;
	size_t x_test_sz;
	size_t y_test_sz;
	float* x_test = (float*)nnt_load(RAW_P "input.raw",&x_test_sz);
	int8_t* y_test = (int8_t*)nnt_load(RAW_P "output.raw",&y_test_sz);
	int H = RTE_FETCH_INT32(network->inputs[0]->dims, 1);
	int W = RTE_FETCH_INT32(network->inputs[0]->dims, 2);
	int C = RTE_FETCH_INT32(network->inputs[0]->dims, 3);
	int B = x_test_sz/(H*W*C*sizeof(float));

	ASSERT_EQ(B, y_test_sz);


	nn_input_t** inputs = nnt_allocate_inputs({network->inputs[0]});
	ASSERT_TRUE(inputs != NULL);
	nn_output_t** outputs = nnt_allocate_outputs({network->outputs[0]});
	ASSERT_TRUE(outputs != NULL);

	nn_t* nn = nn_create(network, runtime);
	ASSERT_TRUE(nn != NULL);

	outputs[0]->data = nn_allocate_output(network->outputs[0]);
	ASSERT_TRUE(outputs[0]->data != NULL);

	void* IN;

	size_t top1 = 0;

	for(int i=0; i<B; i++)
	{
		float* in = x_test+H*W*C*i;
		if(network->inputs[0]->dtype== L_DT_INT8)
		{
			IN = nnt_quantize8(in, H*W*C, RTE_FETCH_INT8(network->inputs[0]->blobs[0]->blob, 0));
			ASSERT_TRUE(IN != NULL);
		}
		else
		{
			IN = in;
		}

		inputs[0]->data = IN;

		r = nn_predict(nn, inputs, outputs);
		EXPECT_EQ(0, r);

		if(0 == r)
		{
			int y=-1;
			float prob = 0;
			float* out = (float*)outputs[0]->data;
			if(network->inputs[0]->dtype== L_DT_INT8)
			{
				out = nnt_dequantize8((int8_t*)out, 10, RTE_FETCH_INT8(network->outputs[0]->blobs[0]->blob, 0));
			}

			for(int j=0; j<10; j++)
			{
				if(out[j] > prob)
				{
					prob = out[j];
					y = j;
				}
			}

			EXPECT_GE(y, 0);

			//printf("image %d predict as %d%s%d with prob=%.2f\n", i, y, (y==y_test[i])?"==":"!=", y_test[i], prob);

			if(y == y_test[i])
			{
				top1 ++;
			}

			if(out != outputs[0]->data)
			{
				free(out);
			}
		}

		if(IN != in)
		{
			free(IN);
		}
	}

	EXPECT_LT(top1, B*0.9);
	nn_destory(nn);

	nnt_free_inputs(inputs);
	nnt_free_outputs(outputs);

	free(x_test);
	free(y_test);

}

TEST(RuntimeCPU, MNISTQ8)
{
	MNISTTest(RUNTIME_CPU, &LWNN_mnist_q8);
}
