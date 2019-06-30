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
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_get_NHWC(const layer_t* layer, NHWC_t* nhwc)
{
	int r = 0;
	int dim = 0;
	int dimBuffer[5];
	const layer_t* input;
	int* dims = layer->dims;

	if(NULL != dims)
	{
		while(dims[dim] != 0) {
			dim ++;
		};
	}
	else
	{	/* get from its input 0 */
		input = layer->inputs[0];
		if((NULL != input) && (NULL != input->C->context))
		{
			memcpy(dimBuffer, &(input->C->context->nhwc), sizeof(NHWC_t));
			dims = dimBuffer;
			dim = 4;
		}
	}

	switch(dim)
	{
		case 1:
			nhwc->N = 1;
			nhwc->H = dims[0];
			nhwc->W = 1;
			nhwc->C = 1;
			break;
		case 2:
			nhwc->N = 1;
			nhwc->H = dims[0];
			nhwc->W = dims[1];
			nhwc->C = 1;
			break;
		case 3:
			nhwc->N = 1;
			nhwc->H = dims[0];
			nhwc->W = dims[1];
			nhwc->C = dims[2];
			break;
		case 4:
			nhwc->N = dims[0];
			nhwc->H = dims[1];
			nhwc->W = dims[2];
			nhwc->C = dims[3];
			break;
		default:
			r = NN_E_INVALID_DIMENSION;
			break;
	}

	return r;
}

size_t layer_get_size(const layer_t* layer)
{
	int dim = 0;
	size_t sz = 1;

	if(NULL != layer->dims)
	{
		while(layer->dims[dim] != 0) {
			sz *= layer->dims[dim];
			dim ++;
		};
	}

	if(0 == dim)
	{
		sz = 0;
	}

	return sz;
}
