/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020 Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "TfLite.hpp"
/* ============================ [ MACROS    ] ====================================================== */
namespace lwnn {
#define kTensorArenaSize 1000000
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static tflite::AllOpsResolver resolver;
static tflite::MicroErrorReporter error_reporter;
/* ============================ [ FUNCTIONS ] ====================================================== */
TfLite::TfLite(std::string binary)
{
	FILE* fb = fopen(binary.c_str(), "rb");
	if(nullptr == fb) {
		throw std::runtime_error("No binary file: " + binary);
	}
	fseek(fb, 0, SEEK_END);
	size_t size = ftell(fb);
	fseek(fb, 0, SEEK_SET);
	m_ModelData = new uint8_t[size];
	fread(m_ModelData, 1, size, fb);
	fclose(fb);

	m_Model = tflite::GetModel(m_ModelData);

	m_TensorArea = new uint8_t[kTensorArenaSize];
	m_Interpreter = new tflite::MicroInterpreter(
			m_Model, resolver, m_TensorArea, kTensorArenaSize, &error_reporter);
}
TfLite::~TfLite()
{
	delete[] m_ModelData;
	delete[] m_TensorArea;
	delete m_Interpreter;
}
py::object TfLite::predict(py::dict feed)
{
	py::dict outputs;

	return outputs;
}
}
