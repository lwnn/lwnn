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
int NHWC_from(const int* dims, NHWC_t* nhwc)
{
	int r = 0;
	int dim = 0;

	assert(dims != NULL);

	while((dims[dim] != 0) && (dim < 4)) {
		dim ++;
	};

	switch(dim)
	{
		case 1:
			nhwc->N = 1;
			nhwc->H = 1;
			nhwc->W = 1;
			nhwc->C = dims[0];
			break;
		case 2:
			nhwc->N = 1;
			nhwc->H = 1;
			nhwc->W = dims[0];
			nhwc->C = dims[1];
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
			NNLOG(NN_ERROR, ("invalid dimension\n"));
			r = NN_E_INVALID_DIMENSION;
			break;
	}

	return r;
}

int layer_get_NHWC(const layer_t* layer, NHWC_t* nhwc)
{
	return NHWC_from(layer->dims, nhwc);
}

size_t layer_get_size(const layer_t* layer)
{
	int dim = 0;
	size_t sz = 1;

	if(NULL != layer->C->context)
	{
		dim = 4;
		sz = NHWC_SIZE(layer->C->context->nhwc);
	}
	else if(NULL != layer->dims)
	{
		while(layer->dims[dim] != 0) {
			sz *= layer->dims[dim];
			dim ++;
		};
	}
	else
	{
		NNLOG(NN_ERROR, ("can't get %s layer size for now\n", layer->name));
	}

	if(0 == dim)
	{
		sz = 0;
	}

	return sz;
}

/* ============================ [ UNSUPPORTED ] ==================================================== */
UNSUPPORTED_LAYER_OPS(cpu_float, RELU)

UNSUPPORTED_LAYER_OPS(cpu_float, MAXPOOL)

UNSUPPORTED_LAYER_OPS(cpu_float, RESHAPE)
UNSUPPORTED_LAYER_OPS(cpu_q8, RESHAPE)
UNSUPPORTED_LAYER_OPS(cl, RESHAPE)

UNSUPPORTED_LAYER_OPS(cpu_float, DENSE)

UNSUPPORTED_LAYER_OPS(cpu_float, SOFTMAX)
