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

	if(NULL != layer->dims)
	{
		while(layer->dims[dim] != 0) {
			dim ++;
		};
	}

	switch(dim)
	{
		case 1:
			nhwc->N = 1;
			nhwc->H = layer->dims[0];
			nhwc->W = 1;
			nhwc->C = 1;
			break;
		case 2:
			nhwc->N = 1;
			nhwc->H = layer->dims[0];
			nhwc->W = layer->dims[1];
			nhwc->C = 1;
			break;
		case 3:
			nhwc->N = 1;
			nhwc->H = layer->dims[0];
			nhwc->W = layer->dims[1];
			nhwc->C = layer->dims[2];
			break;
		case 4:
			nhwc->N = layer->dims[0];
			nhwc->H = layer->dims[1];
			nhwc->W = layer->dims[2];
			nhwc->C = layer->dims[3];
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
