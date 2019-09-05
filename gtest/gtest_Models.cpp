/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
#include "bbox_util.hpp"
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_MNIST_NOT_FOUND_OKAY FALSE
#define NNT_MNIST_TOP1 0.9

#define NNT_UCI_INCEPTION_NOT_FOUND_OKAY TRUE
#define NNT_UCI_INCEPTION_TOP1 0.85

#define NNT_SSD_NOT_FOUND_OKAY TRUE
#define NNT_SSD_TOP1 0.9

/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	void* (*load_input)(const char* path, int id, size_t* sz);
	void* (*load_output)(const char* path, int id, size_t* sz);
	int (*compare)(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg);
	size_t n;
} nnt_model_args_t;
/* ============================ [ DECLARES  ] ====================================================== */
static void* load_input(const char* path, int id, size_t* sz);
static void* load_output(const char* path, int id, size_t* sz);
static int ssd_compare(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg);
/* ============================ [ DATAS     ] ====================================================== */
NNT_CASE_DEF(MNIST) =
{
	NNT_CASE_DESC(mnist),
};

NNT_CASE_DEF(UCI_INCEPTION) =
{
	NNT_CASE_DESC(uci_inception),
};

static const nnt_model_args_t nnt_ssd_args =
{
	load_input,
	load_output,
	ssd_compare,
	7	/* 7 test images */
};

NNT_CASE_DEF(SSD) =
{
	NNT_CASE_DESC_ARGS(ssd),
};
/* ============================ [ LOCALS    ] ====================================================== */
static void* load_input(const char* path, int id, size_t* sz)
{
	char name[256];
	snprintf(name, sizeof(name), "%s/input%d.raw", path, id);

	return nnt_load(name, sz);
}

static void* load_output(const char* path, int id, size_t* sz)
{
	char name[256];
	snprintf(name, sizeof(name), "%s/output%d.raw", path, id);

	return nnt_load(name, sz);
}
static int ssd_compare(nn_t* nn, int id, float* output, size_t szo, float* gloden, size_t szg)
{
	int r = 0;
	int i;
	float IoU;
	int num_det = nn->network->outputs[0]->layer->C->context->nhwc.N;

	EXPECT_EQ(num_det, szg/7);

	for(i=0; i<num_det; i++)
	{
		IoU = ssd::JaccardOverlap(&output[7*i+3], &gloden[7*i+3]);

		EXPECT_EQ(output[7*i], gloden[7*i]);	/* batch */
		EXPECT_EQ(output[7*i+1], gloden[7*i+1]); /* label */
		EXPECT_NEAR(output[7*i+2], gloden[7*i+2], 0.05); /* prop */
		EXPECT_GT(IoU, 0.9);

		if(output[7*i] != gloden[7*i])
		{
			r = -1;
		}

		if(output[7*i+1] != gloden[7*i+1])
		{
			r = -2;
		}

		if(std::fabs(output[7*i+2]-gloden[7*i+2]) > 0.05)
		{
			r = -3;
		}

		if(IoU < 0.9)
		{
			r = -4;
		}
	}

	if(0 != r)
	{
		printf("output for image %d is not correct\n", id);
	}

	return 0;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
void ModelTestMain(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output,
		const nnt_model_args_t* args,
		float mintop1)
{
	int r = 0;
	size_t x_test_sz;
	size_t y_test_sz;
	float* x_test = NULL;
	int32_t* y_test = NULL;

	const nn_input_t* const * inputs = network->inputs;
	const nn_output_t* const * outputs = network->outputs;

	int H,W,C,B;
	int classes;

	nn_t* nn = nn_create(network, runtime);
	ASSERT_TRUE(nn != NULL);

	if(NULL == nn)
	{
		return;
	}

	H = inputs[0]->layer->C->context->nhwc.H;
	W = inputs[0]->layer->C->context->nhwc.W;
	C = inputs[0]->layer->C->context->nhwc.C;
	classes = NHWC_BATCH_SIZE(outputs[0]->layer->C->context->nhwc);
	if(NULL == args)
	{
		x_test = (float*)nnt_load(input, &x_test_sz);
		y_test = (int32_t*)nnt_load(output,&y_test_sz);
		B = x_test_sz/(H*W*C*sizeof(float));
		ASSERT_EQ(B, y_test_sz/sizeof(int32_t));
	}
	else
	{
		B = args->n;
	}

	void* IN;

	size_t top1 = 0;
	for(int i=0; (i<B) && (r==0); i++)
	{
		if(g_CaseNumber != -1)
		{
			i = g_CaseNumber;
		}
		float* in;
		size_t sz_in;
		float* golden;
		size_t sz_golden;

		if(NULL == args)
		{
			in = x_test+H*W*C*i;
		}
		else
		{
			in = (float*)args->load_input(input, i, &sz_in);
			EXPECT_EQ(sz_in, H*W*C*sizeof(float));
			golden = (float*)args->load_output(input, i, &sz_golden);
			ASSERT_TRUE(golden != NULL);
		}

		if(network->type== NETWORK_TYPE_Q8)
		{
			sz_in = H*W*C;
			IN = nnt_quantize8(in, H*W*C, LAYER_Q(inputs[0]->layer));
			ASSERT_TRUE(IN != NULL);
		}
		else if(network->type== NETWORK_TYPE_S8)
		{
			sz_in = H*W*C;
			IN = nnt_quantize8(in, H*W*C, LAYER_Q(inputs[0]->layer),
						LAYER_Z(inputs[0]->layer),
						(float)LAYER_S(inputs[0]->layer)/NN_SCALER);
			ASSERT_TRUE(IN != NULL);
		}
		else if(network->type== NETWORK_TYPE_Q16)
		{
			sz_in = H*W*C*sizeof(int16_t);
			IN = nnt_quantize16(in, H*W*C, LAYER_Q(inputs[0]->layer));
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
			if(network->type== NETWORK_TYPE_Q8)
			{
				out = nnt_dequantize8((int8_t*)out, classes, LAYER_Q(outputs[0]->layer));
			}
			else if(network->type== NETWORK_TYPE_S8)
			{
				out = nnt_dequantize8((int8_t*)out, classes, LAYER_Q(outputs[0]->layer),
						LAYER_Z(outputs[0]->layer),
						(float)LAYER_S(outputs[0]->layer)/NN_SCALER);
			}
			else if(network->type== NETWORK_TYPE_Q16)
			{
				out = nnt_dequantize16((int16_t*)out, classes, LAYER_Q(outputs[0]->layer));
			}

			if(NULL == args)
			{
				for(int j=0; j<classes; j++)
				{
					if(out[j] > prob)
					{
						prob = out[j];
						y = j;
					}
				}

				EXPECT_GE(y, 0);

				if(y == y_test[i])
				{
					top1 ++;
				}
			}
			else
			{
				y = args->compare(nn, i, out, classes, golden, sz_golden/sizeof(float));
				if(0 == y)
				{
					top1 ++;
				}

				free(in);
				free(golden);
			}

			if(g_CaseNumber != -1)
			{
				if(NULL == args)
				{
					printf("image %d predict as %d%s%d with prob=%.2f\n", i, y, (y==y_test[i])?"==":"!=", y_test[i], prob);
				}
				break;
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
			printf("LWNN TOP1 is %f on %d test images\n", (float)top1/i, i);
		}
	}

	if(-1 == g_CaseNumber)
	{
		printf("LWNN TOP1 is %f\n", (float)top1/B);
		EXPECT_GT(top1, B*mintop1);
	}
	nn_destory(nn);

	if(NULL != x_test)
	{
		free(x_test);
	}
	if(NULL != y_test)
	{
		free(y_test);
	}

}

void NNTModelTestGeneral(runtime_type_t runtime,
		const char* netpath,
		const char* input,
		const char* output,
		const void* args,
		float mintop1,
		float not_found_okay)
{
	const network_t* network;
	void* dll;
	network = nnt_load_network(netpath, &dll);
	if(not_found_okay == FALSE)
	{
		EXPECT_TRUE(network != NULL);
	}
	if(network == NULL)
	{
		return;
	}
	ModelTestMain(runtime, network, input, output, (const nnt_model_args_t*)args, mintop1);
	dlclose(dll);
}

NNT_MODEL_TEST_ALL(MNIST)

NNT_MODEL_TEST_ALL(UCI_INCEPTION)

NNT_MODEL_TEST_CPU_FLOAT(SSD)
