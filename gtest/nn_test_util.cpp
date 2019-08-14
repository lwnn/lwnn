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
int nnt_run(const network_t* network,
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
	else
	{
		r = -99;
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

	assert(max_diff > 0.0);

	for(size_t i=0; i<sz; i++)
	{
		float base = std::fabs(A[i]);
		if((base/max_diff) < 100)
		{
			base = max_diff*100;
		}
		float diff = std::fabs(A[i]-B[i])/base;

		if(diff > max_diff)
		{
			if(equal < 8)
			{
				printf("@%d %f != %f\n", i, A[i], B[i]);
			}
			equal++;
		}
	}

	return equal;
}

int8_t* nnt_quantize8(float* in, size_t sz, int8_t Q)
{
	int8_t* out = (int8_t*)malloc(sz);
	assert(out);

	for(size_t i=0; i<sz; i++)
	{
		out[i] = std::round(in[i]*(std::pow(2,Q)));
	}

	return out;
}

float* nnt_dequantize8(int8_t* in , size_t sz, int8_t Q)
{
	float* out = (float*)malloc(sz*sizeof(float));
	assert(out);

	for(size_t i=0; i<sz; i++)
	{
		out[i] = in[i]/(std::pow(2,Q));
	}

	return out;
}

int16_t* nnt_quantize16(float* in, size_t sz, int8_t Q)
{
	int16_t* out = (int16_t*)malloc(sz*sizeof(int16_t));
	assert(out);

	for(size_t i=0; i<sz; i++)
	{
		out[i] = std::round(in[i]*(std::pow(2,Q)));
	}

	return out;
}

float* nnt_dequantize16(int16_t* in , size_t sz, int8_t Q)
{
	float* out = (float*)malloc(sz*sizeof(float));
	assert(out);

	for(size_t i=0; i<sz; i++)
	{
		out[i] = in[i]/(std::pow(2,Q));
	}

	return out;
}

void nnt_siso_network_test(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output,
		float max_diff,
		float qmax_diff)
{
	nn_input_t** inputs = nnt_allocate_inputs({network->inputs[0]});
	nn_output_t** outputs = nnt_allocate_outputs({network->outputs[0]});

	size_t sz_in;
	float* IN = (float*)nnt_load(input, &sz_in);
	ASSERT_EQ(sz_in, layer_get_size((inputs[0])->layer)*sizeof(float));

	int8_t* in8 = NULL;
	int16_t* in16 = NULL;
	if(network->layers[0]->dtype== L_DT_INT8)
	{
		int8_t* blob = (int8_t*)network->layers[0]->blobs[0]->blob;
		in8 = nnt_quantize8(IN, sz_in/sizeof(float), blob[0]);
		memcpy(inputs[0]->data, in8, sz_in/sizeof(float));
	}
	else if(network->layers[0]->dtype== L_DT_INT16)
	{
		int8_t* blob = (int8_t*)network->layers[0]->blobs[0]->blob;
		in16 = nnt_quantize16(IN, sz_in/sizeof(float), blob[0]);
		memcpy(inputs[0]->data, in16, sz_in*sizeof(int16_t)/sizeof(float));
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
					layer_get_size(outputs[0]->layer), max_diff);
			free(out);
			/* if (1-qmax_diff)*100 percent data is okay, pass test */
			EXPECT_LE(r, layer_get_size(outputs[0]->layer)*qmax_diff);
		}
		else if(in16 != NULL)
		{
			int8_t* blob = (int8_t*)outputs[0]->layer->blobs[0]->blob;
			float* out = nnt_dequantize16((int16_t*)outputs[0]->data, layer_get_size(outputs[0]->layer), blob[0]);
			r = nnt_is_equal(OUT, out,
					layer_get_size(outputs[0]->layer), max_diff);
			free(out);
			/* if (1-qmax_diff)*100 percent data is okay, pass test */
			EXPECT_LE(r, layer_get_size(outputs[0]->layer)*qmax_diff);
		}
		else
		{
			r = nnt_is_equal(OUT, (float*)outputs[0]->data,
					layer_get_size(outputs[0]->layer), max_diff);
			EXPECT_EQ(0, r);
		}

		free(OUT);
	}

	if(in8 != NULL)
	{
		free(in8);
	}

	if(in16 != NULL)
	{
		free(in16);
	}

	free(IN);

	nnt_free_inputs(inputs);
	nnt_free_outputs(outputs);
}

const network_t* nnt_load_network(const char* netpath, void** dll)
{
	const network_t* network = NULL;
	char path[256];
	char* bname;
	char* cwd;
	char symbol[128];

	cwd = getcwd(NULL,0);
	assert(cwd != NULL);
	snprintf(path, sizeof(path),"%s/%s",cwd, netpath);
	free(cwd);

	*dll = dlopen(path, RTLD_NOW);
	if((*dll) != NULL)
	{
		bname = basename(path);
		assert(bname != NULL);
		#ifdef _WIN32
		bname[strlen(bname)-4] = 0;
		#else
		bname[strlen(bname)-3] = 0;
		#endif
		snprintf(symbol, sizeof(symbol), "LWNN_%s", bname);
		network = (const network_t*)dlsym(*dll, symbol);
		if(NULL == network)
		{
			printf("failed to lookup symbol %s from %s\n", symbol, path);
			dlclose(*dll);
			*dll = NULL;
		}
	}
	else
	{
		printf("failed to load %s: %s\n", path, dlerror());
	}

	return network;
}

void NNTTestGeneral(runtime_type_t runtime,
		const char* netpath,
		const char* input,
		const char* output,
		float max_diff,
		float qmax_diff)
{
	const network_t* network;
	void* dll;

	network = nnt_load_network(netpath, &dll);
	EXPECT_TRUE(network != NULL);
	if(network == NULL)
	{
		return;
	}

	if(network->layers[0]->dtype== L_DT_INT8)
	{
		nnt_siso_network_test(runtime, network, input, output, max_diff, qmax_diff);
	}
	else if(network->layers[0]->dtype== L_DT_INT16)
	{
		nnt_siso_network_test(runtime, network, input, output, max_diff, qmax_diff);
	}
	else
	{
		nnt_siso_network_test(runtime, network, input, output);
	}

	dlclose(dll);
}

