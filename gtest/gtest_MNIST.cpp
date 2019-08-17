/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
#define MNIST_RAW_P RAW_P "mnist/golden/"
#define MNIST_PATH "build/" RAW_P "mnist/" LIBFIX "mnist_"
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
void MNISTTestMain(runtime_type_t runtime,
		const network_t* network)
{
	int r = 0;
	size_t x_test_sz;
	size_t y_test_sz;
	float* x_test = (float*)nnt_load(MNIST_RAW_P "input.raw",&x_test_sz);
	int8_t* y_test = (int8_t*)nnt_load(MNIST_RAW_P "output.raw",&y_test_sz);

	const nn_input_t* const * inputs = network->inputs;
	const nn_output_t* const * outputs = network->outputs;

	int H = RTE_FETCH_INT32(inputs[0]->layer->dims, 1);
	int W = RTE_FETCH_INT32(inputs[0]->layer->dims, 2);
	int C = RTE_FETCH_INT32(inputs[0]->layer->dims, 3);
	int B = x_test_sz/(H*W*C*sizeof(float));

	ASSERT_EQ(B, y_test_sz);

	nn_t* nn = nn_create(network, runtime);
	ASSERT_TRUE(nn != NULL);

	void* IN;

	size_t top1 = 0;
	for(int i=0; (i<B) && (r==0); i++)
	{
		float* in = x_test+H*W*C*i;
		size_t sz_in;
		if(inputs[0]->layer->dtype== L_DT_INT8)
		{
			sz_in = H*W*C;
			IN = nnt_quantize8(in, H*W*C, RTE_FETCH_INT8(inputs[0]->layer->blobs[0]->blob, 0));
			ASSERT_TRUE(IN != NULL);
		}
		else if(inputs[0]->layer->dtype== L_DT_INT16)
		{
			sz_in = H*W*C*sizeof(int16_t);
			IN = nnt_quantize16(in, H*W*C, RTE_FETCH_INT8(inputs[0]->layer->blobs[0]->blob, 0));
			ASSERT_TRUE(IN != NULL);
		}
		else
		{
			sz_in = H*W*C*sizeof(float);
			IN = in;
		}

		memcpy(inputs[0]->data, IN, sz_in);

		r = nn_predict(nn);
		EXPECT_EQ(0, r);

		if(0 == r)
		{
			int y=-1;
			float prob = 0;
			float* out = (float*)outputs[0]->data;
			if(inputs[0]->layer->dtype== L_DT_INT8)
			{
				out = nnt_dequantize8((int8_t*)out, 10, RTE_FETCH_INT8(outputs[0]->layer->blobs[0]->blob, 0));
			}
			else if(inputs[0]->layer->dtype== L_DT_INT16)
			{
				out = nnt_dequantize16((int16_t*)out, 10, RTE_FETCH_INT8(outputs[0]->layer->blobs[0]->blob, 0));
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

		if((i>0) && ((i%1000) == 0))
		{
			printf("MNIST on LWNN TOP1 is %f on %d test images\n", (float)top1/i, i);
		}
	}

	printf("MNIST on LWNN TOP1 is %f\n", (float)top1/B);

	EXPECT_GT(top1, B*0.9);
	nn_destory(nn);

	free(x_test);
	free(y_test);

}

void MNISTTest(runtime_type_t runtime, const char* netpath)
{
	const network_t* network;
	void* dll;
	network = nnt_load_network(netpath, &dll);
	EXPECT_TRUE(network != NULL);
	if(network == NULL)
	{
		return;
	}
	MNISTTestMain(runtime, network);
	dlclose(dll);
}

TEST(RuntimeCPU, MNISTQ8)
{
	MNISTTest(RUNTIME_CPU, MNIST_PATH "q8" DLLFIX);
}

TEST(RuntimeCPU, MNISTQ16)
{
	MNISTTest(RUNTIME_CPU, MNIST_PATH "q16" DLLFIX);
}


TEST(RuntimeCPU, MNISTFloat)
{
	MNISTTest(RUNTIME_CPU, MNIST_PATH "float" DLLFIX);
}
#ifndef DISABLE_RUNTIME_OPENCL
TEST(RuntimeOPENCL, MNIST)
{
	MNISTTest(RUNTIME_OPENCL, MNIST_PATH "float" DLLFIX);
}
#endif
