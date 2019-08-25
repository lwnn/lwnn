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
int g_CaseNumber = -1;
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int nnt_run(const network_t* network,
			runtime_type_t runtime)
{
	int r = 0;
	nn_t* nn = nn_create(network, runtime);
	EXPECT_TRUE(nn != NULL);

	if(nn != NULL)
	{
		r = nn_predict(nn);
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
	if(NULL==fp)
	{
		printf("failed to load raw %s\n", inraw);
		assert(0);
	}
	fseek(fp, 0, SEEK_END);
	*sz = ftell(fp);
	fseek(fp, 0, SEEK_SET);
	in = malloc(*sz);
	fread(in, 1, *sz, fp);
	fclose(fp);
	return in;
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

int8_t* nnt_quantize8(float* in, size_t sz, int8_t Q, int8_t Z, float scale)
{
	int8_t* out = (int8_t*)malloc(sz);
	float v;
	assert(out);

	for(size_t i=0; i<sz; i++)
	{
		v = std::round(in[i]*scale*(std::pow(2,Q)))-Z;
		if(v > 0x7F)
		{
			out[i] = 0x7F;
		}
		else if(v < -0x80)
		{
			out[i] = -0x80;
		}
		else
		{
			out[i] = v;
		}

	}

	return out;
}

float* nnt_dequantize8(int8_t* in , size_t sz, int8_t Q, int8_t Z, float scale)
{
	float* out = (float*)malloc(sz*sizeof(float));
	assert(out);

	for(size_t i=0; i<sz; i++)
	{
		out[i] = scale*(in[i]+Z)/(std::pow(2,Q));
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
	const nn_input_t* const * inputs = network->inputs;
	const nn_output_t* const * outputs = network->outputs;
	size_t sz_in;

	float* IN = (float*)nnt_load(input, &sz_in);
	ASSERT_EQ(sz_in, layer_get_size((inputs[0])->layer)*sizeof(float));

	int8_t* in8 = NULL;
	int16_t* in16 = NULL;
	if(network->type== NETWORK_TYPE_Q8)
	{
		in8 = nnt_quantize8(IN, sz_in/sizeof(float), LAYER_Q(inputs[0]->layer));
		memcpy(inputs[0]->data, in8, sz_in/sizeof(float));
	}
	else if(network->type== NETWORK_TYPE_S8)
	{
		in8 = nnt_quantize8(IN, sz_in/sizeof(float), LAYER_Q(inputs[0]->layer),
					LAYER_Z(inputs[0]->layer), (float)LAYER_S(inputs[0]->layer)/NN_SCALER);
		memcpy(inputs[0]->data, in8, sz_in/sizeof(float));
	}
	else if(network->type== NETWORK_TYPE_Q16)
	{
		in16 = nnt_quantize16(IN, sz_in/sizeof(float), LAYER_Q(inputs[0]->layer));
		memcpy(inputs[0]->data, in16, sz_in*sizeof(int16_t)/sizeof(float));
	}
	else
	{
		memcpy(inputs[0]->data, IN, sz_in);
	}

	int r = nnt_run(network, runtime);

	if(0 == r)
	{
		size_t sz_out;
		float* OUT = (float*)nnt_load(output, &sz_out);
		ASSERT_EQ(sz_out, layer_get_size((outputs[0])->layer)*sizeof(float));

		if(in8 != NULL)
		{
			float* out;
			if(network->type== NETWORK_TYPE_Q8)
			{
				out = nnt_dequantize8((int8_t*)outputs[0]->data, layer_get_size(outputs[0]->layer), LAYER_Q(outputs[0]->layer));
			}
			else
			{
				int32_t* blob = (int32_t*)outputs[0]->layer->blobs[0]->blob;
				out = nnt_dequantize8((int8_t*)outputs[0]->data, layer_get_size(outputs[0]->layer),
						LAYER_Q(outputs[0]->layer), LAYER_Z(outputs[0]->layer), (float)LAYER_S(outputs[0]->layer)/NN_SCALER);
			}
			r = nnt_is_equal(OUT, out,
					layer_get_size(outputs[0]->layer), max_diff);
			free(out);
			/* if (1-qmax_diff)*100 percent data is okay, pass test */
			EXPECT_LE(r, layer_get_size(outputs[0]->layer)*qmax_diff);
		}
		else if(in16 != NULL)
		{
			float* out = nnt_dequantize16((int16_t*)outputs[0]->data, layer_get_size(outputs[0]->layer), LAYER_Q(outputs[0]->layer));
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
		#ifdef _WIN32
		snprintf(symbol, sizeof(symbol), "LWNN_%s", bname);
		#else
		snprintf(symbol, sizeof(symbol), "LWNN_%s", &bname[3]);
		#endif
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
	printf("  Test %s\n", network->name);
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

