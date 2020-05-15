/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */

#ifndef NN_PYTHON_MODEL_HPP_
#define NN_PYTHON_MODEL_HPP_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include "algorithm.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <string>
#include <dlfcn.h>
#include <unistd.h>
namespace py = pybind11;
/* ============================ [ MACROS    ] ====================================================== */
namespace lwnn {
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
class Model {
private:
	void* m_Dll = nullptr;
	const network_t* m_Network= nullptr;
	nn_t* nn = nullptr;

public:
	Model(int runtime, std::string symbol, std::string library, std::string binary);
	~Model();
	py::object predict(py::list outputs, py::dict feed);

public:
#ifndef DISABLE_RUNTIME_CPU
	static const int m_RUNTIME_CPU;
#endif
#ifndef DISABLE_RUNTIME_OPENCL
	static const int m_RUNTIME_OPENCL;
#endif
};
}
#endif /* NN_PYTHON_MODEL_HPP_ */
