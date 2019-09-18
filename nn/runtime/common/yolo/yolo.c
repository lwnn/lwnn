/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#if !defined(DISABLE_RUNTIME_CPU_FLOAT) || !defined(DISABLE_RUNTIME_OPENCL)
#include "yolo.h"
#include "algorithm.h"
#include <math.h>
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static void activate_array(float *x, const int n)
{
	int i;
	for(i = 0; i < n; ++i){
		x[i] = logistic_activate(x[i]);
	}
}
static int entry_index(NHWC_t *inhwc, int classes, int batch, int location, int entry)
{
	int n =   location / (inhwc->W*inhwc->H);
	int loc = location % (inhwc->W*inhwc->H);
	return batch*NHWC_BATCH_SIZE(*inhwc) + n*inhwc->W*inhwc->H*(4+classes+1) + entry*inhwc->W*inhwc->H + loc;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int yolo_forward(float* output, const float* input, NHWC_t *inhwc, int num, int classes)
{
	int r = 0;
	int b,n;

	r = alg_transpose(output, input, inhwc, sizeof(float), ALG_TRANSPOSE_FROM_NHWC_TO_NCHW);

	for (b = 0; b < inhwc->N; ++b) {
		for(n = 0; n < num; ++n){
			int index = entry_index(inhwc, classes, b, n*inhwc->W*inhwc->H, 0);
			activate_array(output + index, 2*inhwc->W*inhwc->H);
			index = entry_index(inhwc, classes, b, n*inhwc->W*inhwc->H, 4);
			activate_array(output + index, (1+classes)*inhwc->W*inhwc->H);
		}
	}

	return r;
}
#endif /* DISABLE_RUNTIME_CPU_FLOAT/DISABLE_RUNTIME_OPENCL */
