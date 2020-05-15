/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "Model.hpp"
/* ============================ [ MACROS    ] ====================================================== */
namespace lwnn {
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
#ifndef DISABLE_RUNTIME_CPU
const int Model::m_RUNTIME_CPU = (int)RUNTIME_CPU;
#endif
#ifndef DISABLE_RUNTIME_OPENCL
const int Model::m_RUNTIME_OPENCL = (int)RUNTIME_OPENCL;
#endif
/* ============================ [ LOCALS    ] ====================================================== */
#ifndef L_BLOB_NOT_BUILTIN
#error L_BLOB_NOT_BUILTIN must be defined
#endif
int nn_blob_loader(void* provider, void* saver, size_t size)
{
	int r = 0;
	FILE* fp = (FILE*)provider;

	size_t readB = fread(saver, 1, size, fp);

	if(readB != size) {
		r = -1;
	}

	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
Model::Model(int runtime, std::string symbol, std::string library, std::string binary)
{
	NNLOG(NN_DEBUG, ("Model(%d, %s, %s)\n", runtime, library.c_str(), binary.c_str()));

	if( ! ( false
#ifndef DISABLE_RUNTIME_CPU
			|| ((int)RUNTIME_CPU == runtime)
#endif
#ifndef DISABLE_RUNTIME_OPENCL
			|| ((int)RUNTIME_OPENCL == runtime)
#endif
	)) {
		throw std::invalid_argument("Invalid runtime");
	}

	m_Dll = dlopen(library.c_str(), RTLD_NOW);
	if(nullptr == m_Dll) {
		throw std::runtime_error("No network file: " + library);
	}

	m_Network = (const network_t*)dlsym(m_Dll, symbol.c_str());
	if(nullptr == m_Dll) {
		throw std::runtime_error("No symbol: " + symbol);
	}

	if(binary.size() > 0) {
		FILE* fb = fopen(binary.c_str(), "rb");
		if(nullptr == fb) {
			throw std::runtime_error("No binary file: " + binary);
		}

		int r = nn_load(m_Network, nn_blob_loader, (void*) fb);
		if(0 != r) {
			throw std::runtime_error("Invalid binary file: " + binary);
		}
		fclose(fb);
	}

	nn = nn_create(m_Network, (runtime_type_t)runtime);
	if(nullptr == nn) {
		throw std::runtime_error("Failed to create network: " + symbol);
	}
}

Model::~Model()
{
	if(m_Dll) {
		dlclose(m_Dll);
	}

	if(nn) {
		nn_destory(nn);
	}
}

py::object Model::predict(py::list outputs, py::dict feed)
{
	py::print(outputs);
	py::print(feed);
	return py::none();
}
}
