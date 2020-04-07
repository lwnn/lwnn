/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020 Parai Wang <parai@foxmail.com>
 */
#ifndef NN_RUNTIME_HALIDE_GENOPS_GENCOMMON_H_
#define NN_RUNTIME_HALIDE_GENOPS_GENCOMMON_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include <Halide.h>
/* ============================ [ MACROS    ] ====================================================== */
using namespace Halide;
#ifdef LAYER_HALIDE_CONTEXT_MEMBER
#define LWNN_ON_HALIDE
#endif

#ifndef LWNN_ON_HALIDE
#define HL_INPUT_BUFFER(name, ndim) Input<Buffer<float>> name{#name, ndim};
#define HL_OUTPUT_BUFFER(name, ndim) Output<Buffer<float>> name{#name, ndim};
#define HL_INPUT_INT(name) Input<int> name{#name};
#define HL_FARTHER(name) : public Halide::Generator<name>
#define HL_REGISTER_GENERATOR(layer, fname) HALIDE_REGISTER_GENERATOR(layer, fname)
#else
#define HL_INPUT_BUFFER(name, ndim) Halide::Func name{#name}
#define HL_OUTPUT_BUFFER(name, ndim) Halide::Func name{#name}
#define HL_INPUT_INT(name) int name
#define HL_FARTHER(name)
#define HL_REGISTER_GENERATOR(layer, fname)
#endif
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
#endif /* NN_RUNTIME_HALIDE_GENOPS_GENCOMMON_H_ */
