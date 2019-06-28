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

int nn_predict(const nn_t* nn)
{
	return runtime_execute(nn);
}

void nn_destory(const nn_t* nn)
{
	runtime_destory(nn);
	free((void*)nn);
}

