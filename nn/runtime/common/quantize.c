/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "quantize.h"
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
void dequantize_q8(float* out, int8_t* in, size_t n, int32_t Q)
{
	size_t i;

	for(i=0; i<n; i++)
	{
		out[i] = (float)in[i]/(1<<Q);
	}
}

void dequantize_s8(float* out, int8_t* in, size_t n, int32_t Q, int32_t S, int32_t Z)
{
	size_t i;

	for(i=0; i<n; i++)
	{
		out[i] = (float)S*((float)in[i]+Z)/(1<<Q)/NN_SCALER;
	}
}
void dequantize_q16(float* out, int16_t* in, size_t n, int32_t Q)
{
	size_t i;

	for(i=0; i<n; i++)
	{
		out[i] = (float)in[i]/(1<<Q);
	}
}
