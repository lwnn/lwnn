/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_RUNTIME_COMMON_QUANTIZE_H_
#define NN_RUNTIME_COMMON_QUANTIZE_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include <stdint.h>
/* ============================ [ MACROS    ] ====================================================== */
#ifdef __cplusplus
extern "C" {
#endif
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
void dequantize_q8(float* out, int8_t* in, size_t n, int32_t Q);
void dequantize_s8(float* out, int8_t* in, size_t n, int32_t Q, int32_t S, int32_t Z);
void dequantize_q16(float* out, int16_t* in, size_t n, int32_t Q);
#ifdef __cplusplus
}
#endif
#endif /* NN_RUNTIME_COMMON_QUANTIZE_H_ */
