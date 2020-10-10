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
class TfBuffer {
private:
	void* m_Data = nullptr;
	void* m_ArrayData = nullptr;
	size_t m_Size = 0;
	size_t m_ItemSize = 0;
	size_t m_ArrayItemSize = 0;
	bool m_DataMalloced = false;
	TfLiteTensor* m_Tensor = nullptr;
	size_t m_NDim = 0;
	int m_LayerSize = 0;

public:
	/* Constructor for input buffer */
	TfBuffer(TfLiteTensor* tensor, py::array& array);
	void reload(py::array& array);
	void* data() { return m_Data; }
	size_t size() { return m_Size; }
	void* array_data() { return m_ArrayData; }
	bool is_data_malloced() { return m_DataMalloced; };
	bool need_quantize() { return (m_ArrayItemSize != m_ItemSize); };
	template<typename T> void quantize(T* out, float* in);
	template<typename T> void dequantize(float* out, T* in);
	/* Constructor for output buffer */
	TfBuffer(TfLiteTensor* tensor);
	/* For numpy tensor */
	TfBuffer(py::array& array);
	py::array numpy();
	~TfBuffer();

private:
	template<typename T> void copy2(void* to, py::buffer_info &binfo);
	template<typename T> void copy3(void* to, py::buffer_info &binfo);
	template<typename T> void copy4(void* to, py::buffer_info &binfo);
	template<typename T> void copy(void* to, py::buffer_info &binfo);
	void validate(py::buffer_info &binfo);
	void load(py::array& array);
};

class TfLite {
private:
	uint8_t* m_ModelData = nullptr;
	const tflite::Model* m_Model = nullptr;
	uint8_t* m_TensorArea = nullptr;
	tflite::MicroInterpreter* m_Interpreter = nullptr;
	std::map<TfLiteTensor*, TfBuffer*> m_Buffers;
public:
	TfLite(std::string binary);
	~TfLite();
	py::object predict(py::dict feed);
	void populate_inputs(py::dict& feed);
	void populate_outputs(py::dict& outputs);
};
}
#endif /* NN_PYTHON_TFLITE_HPP_ */
