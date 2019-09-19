/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef _YOLO_YOLO_H_
#define _YOLO_YOLO_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int yolo_forward(float* output, const float* input, NHWC_t *inhwc, int num, int classes);
int yolo_output_forward(const nn_t* nn, const layer_t* layer, void* (*fetch_input)(const nn_t*, const layer_t*));
#endif /* _YOLO_YOLO_H_ */
