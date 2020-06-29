/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_ALGORITHM_H_
#define NN_ALGORITHM_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
#ifdef __cplusplus
extern "C" {
#endif
/* adjust pointer address by offset */
#define APABO(addr, offset) ((void*)(((size_t)(addr))+(offset)))

#define ALG_MAX(o, a, b) if( (a) > (b) ) { (o) = (a); } else { (o) = (b); }
#define ALG_MIN(o, a, b) if( (a) > (b) ) { (o) = (b); } else { (o) = (a); }
#define ALG_ADD(o, a, b) (o) = (a) + (b);
#define ALG_SUB(o, a, b) (o) = (a) - (b);
#define ALG_MUL(o, a, b) (o) = (a) * (b);
#define ALG_POW(o, a, b) (o) = pow(a, b);

#define DEF_ALG_ELTWISE(DT, OP)										\
	void alg_eltwise_##OP##_##DT(DT* A, DT* B, DT* O, size_t sz)	\
	{																\
		size_t i;													\
		for(i=0; i<sz; i++)											\
		{															\
			ALG_##OP(O[i], A[i], B[i])								\
		}															\
	}

#define DEF_ALG_BROADCAST_ONE(DT, OP)									\
	void alg_broadcast_one_##OP##_##DT(DT* A, DT B, DT* O, size_t sz)	\
	{																	\
		size_t i;														\
		for(i=0; i<sz; i++)												\
		{																\
			ALG_##OP(O[i], A[i], B)										\
		}																\
	}

#define DEF_ALG_BROADCAST_CHANNEL(DT, OP)								\
	void alg_broadcast_channel_##OP##_##DT(DT* A, DT* B, DT* O, size_t sz, size_t C)	\
	{																	\
		size_t i,c;														\
		sz = sz/C;														\
		for(i=0; i<sz; i++)												\
		{																\
			for(c=0; c<C; c++)											\
			{															\
				ALG_##OP(O[i*C+c], A[i*C+c], B[c])						\
			}															\
		}																\
	}

#define DEF_ALG_BROADCAST(DT, OP)												\
	void alg_broadcast_##OP##_##DT(DT* A, DT* B, DT* O, NHWC_t* a_nhwc, NHWC_t* b_nhwc)	\
	{																	\
		size_t n,h,w,c;													\
		size_t n_,h_,w_,c_;												\
		size_t offset, offset_;											\
		for(n=0; n<a_nhwc->N; n++) {									\
			ALG_MIN(n_, n, b_nhwc->N-1);								\
			for(h=0; h<a_nhwc->H; h++) {								\
				ALG_MIN(h_, h, b_nhwc->H-1);							\
				for(w=0; w<a_nhwc->W; w++) {							\
					ALG_MIN(w_, w, b_nhwc->W-1);						\
					for(c=0; c<a_nhwc->C; c++) {						\
						ALG_MIN(c_, c, b_nhwc->C-1);					\
						offset = ((n*a_nhwc->H+h)*a_nhwc->W+w)*a_nhwc->C+c;			\
						offset_ = ((n_*b_nhwc->H+h_)*b_nhwc->W+w_)*b_nhwc->C+c_;	\
						ALG_##OP(O[offset], A[offset], B[offset_])		\
					}													\
				}														\
			}															\
		}																\
	}
/* ============================ [ TYPES     ] ====================================================== */
typedef enum
{
	ALG_TRANSPOSE_FROM_NHWC_TO_NCHW=0,
	ALG_TRANSPOSE_FROM_NHWC_TO_NWHC=0x1000,
	ALG_TRANSPOSE_FROM_NHWC_TO_NHCW=0x2000,
	ALG_TRANSPOSE_FROM_NHWC_TO_NCWH=0x2000,
	ALG_TRANSPOSE_FROM_NCHW_TO_NHWC=0x8000,
} alg_transpose_t;

typedef enum{
	ALG_BROADCAST_NONE=0,
	ALG_BROADCAST_ONE=0x1000,
	ALG_BROADCAST_CHANNEL=0x2000,
	ALG_BROADCAST_GENERAL=0x3000,
} alg_broadcast_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int alg_concat(const nn_t* nn, const layer_t* layer, int axis,
		void* pout, void* (*fetch_input)(const nn_t* nn, const layer_t* layer),
		size_t type_size);
int alg_up_sampling(void* pout, void* pin, NHWC_t *outNHWC, NHWC_t *inNHWC, size_t type_size, uint8_t* pmask);
int alg_transpose(void* output, const void* input, const NHWC_t *nhwc, size_t type_size, alg_transpose_t transpose);

int alg_deconv2d_calculate_position(
		int pos,
		int stride,
		int padding,
		int dim_kernel,
		int dim_in,
		int* in_start,
		int* kernel_start,
		int* kernel_end);
int alg_deconv2d_calculate_padding(int dim_kernel, int stride, int dim_in, int dim_out);

int alg_broadcast_prepare(layer_context_t** inputA_context, layer_context_t** inputB_context, alg_broadcast_t *broadcast);
#ifdef __cplusplus
}
#endif
#endif /* NN_ALGORITHM_H_ */
