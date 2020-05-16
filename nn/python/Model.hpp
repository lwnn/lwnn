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
#include <map>
#include <dlfcn.h>
#include <unistd.h>
#include <memory>
#include <stdexcept>
namespace py = pybind11;
/* ============================ [ MACROS    ] ====================================================== */
namespace lwnn {
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
template<typename ... Args>
std::string string_format( const std::string& format, Args ... args )
{
    size_t size = snprintf( nullptr, 0, format.c_str(), args ... ) + 1; /* Extra space for '\0' */
    if( size <= 0 ) { throw std::runtime_error( "Error during formatting." ); }
    std::unique_ptr<char[]> buf( new char[ size ] );
    snprintf( buf.get(), size, format.c_str(), args ... );
    return std::string( buf.get(), buf.get() + size - 1 ); /* We don't want the '\0' inside */
}
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
class Buffer {
private:
	void* m_Data = nullptr;
	void* m_ArrayData = nullptr;
	size_t m_Size;
	size_t m_ItemSize;
	size_t m_ArrayItemSize;
	bool m_DataMalloced = false;
	const layer_t* m_Layer = nullptr;
	size_t m_NDim;
	int m_LayerSize;

public:
	Buffer(void* data_ptr, const layer_t* layer, py::array& array);
	~Buffer() { if(m_DataMalloced) free(m_Data); };
	void reload(py::array& array);
	void* get_data() { return m_Data; }
	void* get_array_data() { return m_ArrayData; }
	bool is_data_malloced() { return m_DataMalloced; };
	bool need_quantize() { return (m_ArrayItemSize != m_ItemSize); };

private:
	template<typename T> void copy2(void* to, py::buffer_info &binfo);
	template<typename T> void copy3(void* to, py::buffer_info &binfo);
	template<typename T> void copy4(void* to, py::buffer_info &binfo);
	template<typename T> void copy(void* to, py::buffer_info &binfo);
	void validate(py::buffer_info &binfo);
	void load(py::array& array);
};

class Model {
private:
	void* m_Dll = nullptr;
	const network_t* m_Network= nullptr;
	nn_t* nn = nullptr;
	std::map<const layer_t*, Buffer*> m_Buffers;

private:
	void populate_inputs(py::dict& feed);

public:
	Model(int runtime, std::string symbol, std::string library, std::string binary);
	~Model();
	py::object predict(py::dict feed);

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
