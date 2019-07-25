/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */

/* ============================ [ FUNCTIONS ] ====================================================== */
int nnt_run(const layer_t* const* network,
			runtime_type_t runtime,
			nn_input_t** inputs,
			nn_output_t** outputs)
{
	int r = 0;
	nn_t* nn = nn_create(network, runtime);
	EXPECT_TRUE(nn != NULL);

	if(nn != NULL)
	{
		for(nn_output_t** o=outputs; (*o) != NULL; o++)
		{
			if(NULL == (*o)->data)
			{
				(*o)->data = nn_allocate_output((*o)->layer);
				assert(NULL != (*o)->data);
			}
		}

		r = nn_predict(nn, inputs, outputs);
		EXPECT_EQ(0, r);
		nn_destory(nn);
	}

	return r;
}

void nnt_fill_inputs_with_random(nn_input_t** inputs, float lo, float hi)
{
	for(nn_input_t** in=inputs; (*in) != NULL; in++)
	{
		size_t sz = layer_get_size((*in)->layer);
		if(L_DT_FLOAT == (*in)->layer->dtype)
		{
			float* data = (float*) (*in)->data;
			for(size_t i=0; i<sz; i++)
			{
				data[i] = lo + static_cast <float> (std::rand()) /( static_cast <float> (RAND_MAX/(hi-lo)));;
			}
		}
		else if(L_DT_INT8 == (*in)->layer->dtype)
		{
			int8_t* data = (int8_t*) (*in)->data;
			for(size_t i=0; i<sz; i++)
			{
				data[i] = lo + static_cast <int> (std::rand()) /( static_cast <int> (RAND_MAX/(hi-lo)));;
			}
		}
		else
		{
			assert(0);
		}
	}
}

void* nnt_load(const char* inraw, size_t *sz)
{
	void* in;

	FILE* fp = fopen(inraw,"rb");
	assert(fp);
	fseek(fp, 0, SEEK_END);
	*sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	in = malloc(*sz);
	fread(in, 1, *sz, fp);
	fclose(fp);
	return in;
}

nn_input_t** nnt_allocate_inputs(std::vector<const layer_t*> layers)
{
	int sz = layers.size();
	nn_input_t* inputs = new nn_input_t[sz];
	nn_input_t** inputs_list = new nn_input_t*[sz+1];

	assert(NULL != inputs);
	assert(NULL != inputs_list);

	for(int i=0; i<sz; i++)
	{
		inputs[i].layer = layers[i];
		inputs[i].data = nn_allocate_input(layers[i]);
		assert(NULL != inputs[i].data);
		inputs_list[i] = &inputs[i];
	}

	inputs_list[sz] = NULL;

	return inputs_list;
}

nn_output_t** nnt_allocate_outputs(std::vector<const layer_t*> layers)
{
	int sz = layers.size();
	nn_output_t* outputs = new nn_output_t[sz];
	nn_output_t** outputs_list = new nn_output_t*[sz+1];

	assert(NULL != outputs);
	assert(NULL != outputs_list);

	for(int i=0; i<sz; i++)
	{
		outputs[i].layer = layers[i];
		outputs[i].data = NULL;
		outputs_list[i] = &outputs[i];
	}

	outputs_list[sz] = NULL;

	return outputs_list;
}

void nnt_free_inputs(nn_input_t** inputs)
{
	for(nn_input_t** in=inputs; (*in) != NULL; in++)
	{
		nn_free_input((*in)->data);
	}

	delete inputs[0];
	delete inputs;
}

void nnt_free_outputs(nn_output_t** ouputs)
{
	for(nn_output_t** o=ouputs; (*o) != NULL; o++)
	{
		nn_free_output((*o)->data);
	}

	delete ouputs[0];
	delete ouputs;
}

int nnt_is_equal(const float* A, const float* B, size_t sz, const float max_diff)
{
	int equal = 0;

	for(size_t i=0; i<sz; i++)
	{
		if(std::fabs(A[i]-B[i]) > max_diff)
		{
			equal++;
			printf("@%d %f != %f\n", i, A[i], B[i]);
		}
	}

	return equal;
}

