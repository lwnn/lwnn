/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020 Parai Wang <parai@foxmail.com>
 */
#ifndef NN_RUNTIME_HALIDE_GENOPS_GENCOMMON_H_
#define NN_RUNTIME_HALIDE_GENOPS_GENCOMMON_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include <Halide.h>
#include <stdio.h>
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
#define HL_REGISTER_GENERATOR(name) HALIDE_REGISTER_GENERATOR(name##Layer, HL_##name)
#define HL_GET_TARGET_ARCH() get_target().arch
#else
#define HL_INPUT_BUFFER(name, ndim) Halide::Func name{#name}
#define HL_OUTPUT_BUFFER(name, ndim) Halide::Func name{#name}
#define HL_INPUT_INT(name) int name
#define HL_FARTHER(name)
#define HL_REGISTER_GENERATOR(name)
#define HL_GET_TARGET_ARCH() Target::X86
#endif
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
#endif /* NN_RUNTIME_HALIDE_GENOPS_GENCOMMON_H_ */
