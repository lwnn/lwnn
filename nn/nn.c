/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */

/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
int nn_log_level = NN_INFO;
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
nn_t* nn_create(const layer_t* const* network, runtime_type_t runtime_type)
{
	nn_t* nn;

	nn = malloc(sizeof(nn_t));
	if(NULL != nn)
	{
		nn->runtime_type = runtime_type;
		nn->network = network;

		nn->runtime = runtime_create(nn);
	}

	if(NULL != nn->runtime)
	{
		int r = runtime_init(nn);

		if(0 != r)
		{
			runtime_destory(nn);
			nn->runtime = NULL;
		}
	}

	if(NULL == nn->runtime)
	{
		free(nn);
		nn = NULL;
	}

	return nn;
}

void nn_set_log_level(int level)
{
	nn_log_level = level;
}

int nn_predict(nn_t* nn, const nn_input_t* const * inputs)
{
	nn->inputs = inputs;
	return runtime_execute(nn);
}

void* nn_get_input_data(const nn_t* nn, const layer_t* layer)
{
	void* data = NULL;

	const nn_input_t* const* input = nn->inputs;

	while(((*input) != NULL) && (NULL == data))
	{
		if((*input)->layer == layer)
		{
			data = (*input)->data;
		}

		input++;
	}

	return data;
}

void nn_destory(nn_t* nn)
{
	if(NULL != nn)
	{
		runtime_destory(nn);
		free(nn);
	}
}

void* nn_allocate_input(const layer_t* layer)
{
	void* mem;
	size_t sz = layer_get_size(layer);
	switch(layer->dtype)
	{
		case L_DT_INT8:
		case L_DT_UINT8:
			break;
		case L_DT_INT16:
		case L_DT_UINT16:
			sz *= 2;
			break;
		case L_DT_INT32:
		case L_DT_UINT32:
		case L_DT_FLOAT:
			sz *= 4;
			break;
		default:
			NNLOG(NN_ERROR,("invalid dtype(%d) for input %s\n",
					layer->dtype, layer->name));
			sz = 0;
			break;
	}

	if(sz > 0)
	{
		mem = malloc(sz);
	}

	return mem;
}


void nn_free_input(void* input)
{
	free(input);
}
