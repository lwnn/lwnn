/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */

#ifndef NN_PYTHON_TFLITE_HPP_
#define NN_PYTHON_TFLITE_HPP_
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include "quantize.h"
#include "algorithm.h"
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <math.h>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <dlfcn.h>
#include <unistd.h>
#include <memory>
#include <stdexcept>
#include "Model.hpp"
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"
namespace py = pybind11;
/* ============================ [ MACROS    ] ====================================================== */
namespace lwnn {
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */

/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
class TfLite {
private:
	uint8_t* m_ModelData = nullptr;
	const tflite::Model* m_Model = nullptr;
	uint8_t* m_TensorArea = nullptr;
	tflite::MicroInterpreter* m_Interpreter = nullptr;
public:
	TfLite(std::string binary);
	~TfLite();
	py::object predict(py::dict feed);
};
}
#endif /* NN_PYTHON_TFLITE_HPP_ */
