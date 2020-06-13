/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef DISABLE_RUNTIME_CPU
RTE_DEF(CPU)
#endif
#ifndef DISABLE_RUNTIME_OPENCL
RTE_DEF(OPENCL)
#endif
#ifdef ENABLE_RUNTIME_HALIDE
RTE_DEF(HALIDE)
#endif
#ifdef ENABLE_RUNTIME_ANY
RTE_DEF(ANY)
#endif
