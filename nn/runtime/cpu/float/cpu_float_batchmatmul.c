/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
} layer_cpu_float_batchmatmul_context_t;

typedef void (*batchmatmul_ref_t)(const float*, const float*, const int, const int, const int, const int, float*);
/* ============================ [ DECLARES  ] ====================================================== */
static void batchmatmul_ref1(const float * A, /* A [H, M, K] */
							const float * B,  /* B [H, N, K] */
							const int H,
							const int M,
							const int K,
							const int N,
							float * O)		 /* O [H, M, N] */
{
	float sum;
	int h,m,k,n;
	for(h=0; h<H; h++) {
		for(m=0; m<M; m++) {
			for(n=0; n<N; n++) {
				sum = 0;
				for(k=0; k<K; k++) {
					sum += A[(h*M+m)*K + k] * B[(h*N+n)*K + k];
				}
				O[(h*M+m)*N + n] = sum;
			}
		}
	}
}

static void batchmatmul_ref2(const float * A, /* A [H, M, K] */
							const float * B,  /* B [H, K, N] */
							const int H,
							const int M,
							const int K,
							const int N,
							float * O)		/* O [H, M, N] */
{
	float sum;
	int h,m,k,n;
	for(h=0; h<H; h++) {
		for(m=0; m<M; m++) {
			for(n=0; n<N; n++) {
				sum = 0;
				for(k=0; k<K; k++) {
					sum += A[(h*M+m)*K + k] * B[(h*K+k)*N + n];
				}
				O[(h*M+m)*N + n] = sum;
			}
		}
	}
}
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_BATCHMATMUL_init(const nn_t* nn, const layer_t* layer)
{
	return rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_batchmatmul_context_t), sizeof(float));
}

int layer_cpu_float_BATCHMATMUL_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_batchmatmul_context_t* context = (layer_cpu_float_batchmatmul_context_t*)layer->C->context;
	layer_cpu_context_t* inputA_context = (layer_cpu_context_t*)layer->inputs[0]->C->context;
	layer_cpu_context_t* inputB_context = (layer_cpu_context_t*)layer->inputs[1]->C->context;
	float *A = (float*)inputA_context->out[0];
	float *B = (float*)inputB_context->out[0];
	float *O = (float*)context->out[0];
	batchmatmul_ref_t batchmatmul_ref;

	size_t batch;
	size_t batch_sizeA = NHWC_BATCH_SIZE(inputA_context->nhwc);
	size_t batch_sizeB = NHWC_BATCH_SIZE(inputB_context->nhwc);
	size_t batch_sizeO = NHWC_BATCH_SIZE(context->nhwc);

	int M = context->nhwc.W;
	int K = inputA_context->nhwc.C;
	int N = context->nhwc.C;
	int H = context->nhwc.H;

	assert(inputA_context->nhwc.N == inputB_context->nhwc.N);
	assert(H == inputA_context->nhwc.H);
	assert(H == inputB_context->nhwc.H);
	assert(M == inputA_context->nhwc.W);
	assert(K == inputA_context->nhwc.C);

	/* TODO: the right way is to check the attribute of transposeA and transposeB */
	if(K == inputB_context->nhwc.C) {
		/* assert A in shape [B,H,M,K], B in shape [B,H,N,K], so O in shape [B,H,M,N] */
		batchmatmul_ref = batchmatmul_ref1;
		assert(N == inputB_context->nhwc.W);
	} else {
		/* assert A in shape [B,H,M,K], B in shape [B,H,K,N], so O in shape [B,H,M,N] */
		batchmatmul_ref = batchmatmul_ref2;
		assert(N == inputB_context->nhwc.C);
		assert(K == inputB_context->nhwc.W);
	}

	rte_cpu_dynamic_batch(layer, inputA_context);

	NNLOG(NN_DEBUG, (" H=%d, M,K,N=%d,%d,%d\n", H, M, K, N));

	for(batch=0; batch<context->nhwc.N; batch++) {
		batchmatmul_ref(A+batch_sizeA*batch,
						B+batch_sizeB*batch,
						H,M,K,N,
						O+batch_sizeO*batch);
	}

	return r;
}

void layer_cpu_float_BATCHMATMUL_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
