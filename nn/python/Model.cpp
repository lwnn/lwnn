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
Buffer::Buffer(void* data_ptr, const layer_t* layer, py::array& array)
		: m_Data(data_ptr), m_Layer(layer)
{
	py::buffer_info binfo = array.request();

	m_Size = binfo.size;
	m_ArrayItemSize = binfo.itemsize;
	switch(m_Layer->dtype) {
		case L_DT_INT8: m_ItemSize = 1; break;
		case L_DT_INT16: m_ItemSize = 2; break;
		case L_DT_FLOAT: m_ItemSize = 4; break;
		default: throw std::runtime_error("invalid layer data type"); break;
	}
	m_LayerSize = NHWC_SIZE(layer->C->context->nhwc);
	for(m_NDim=0; m_Layer->dims[m_NDim] != 0; m_NDim++);
	validate(binfo);

	if(nullptr == m_Data) {
		m_Data = malloc(m_Size*m_ItemSize);
		if(nullptr == m_Data) {
			throw std::runtime_error("No memory for m_Data");
		}
		m_DataMalloced = true;
	}

	load(array);
};

void Buffer::validate(py::buffer_info &binfo)
{
	bool valid = true;

	if(m_ArrayItemSize != binfo.itemsize) {
		throw std::runtime_error("invalid data type");
	}

	if(m_ArrayItemSize != m_ItemSize) {
		if( (m_ItemSize!=2) && (m_ItemSize !=1) ) {
			throw std::runtime_error("only support Q8 and Q16");
		}

		if(m_ArrayItemSize != 4) {
			throw std::runtime_error(string_format("need float input for Q%d", m_ItemSize*8));
		}
	}

	if((m_LayerSize > 0) && ((size_t)m_LayerSize != m_Size)) {
		valid = false;
	} else if(m_NDim != binfo.ndim) {
		valid = false;
	} else {
		for(int i=0; i<m_NDim; i++) {
			if(m_Layer->dims[i] != -1) {
				if(m_Layer->dims[i] != binfo.shape[i]) {
					valid = false;
				}
			}
		}
	}

	if(false == valid) {
		char msg[256];
		int len = snprintf(msg, sizeof(msg), "Invalid buffer for %s: required %d (", m_Layer->name, m_LayerSize);
		for(int i=0; i<m_NDim; i++) {
			len += snprintf(&msg[len], sizeof(msg)-len, "%d,", m_Layer->dims[i]);
		}
		len += snprintf(&msg[len], sizeof(msg)-len, "), provided %d (", (int)m_Size);
		for(int i=0; i<binfo.ndim; i++) {
			len += snprintf(&msg[len], sizeof(msg)-len, "%d,", (int)binfo.shape[i]);
		}
		len += snprintf(&msg[len], sizeof(msg)-len, ")");
		throw std::runtime_error(msg);
	}
}

template<typename T> void Buffer::copy2(void* to, py::buffer_info &binfo)
{
	T* _to = (T*) to;
	T* _from = (T*) binfo.ptr;
	int B = binfo.shape[0];
	int H = binfo.shape[1];

	int Bs = binfo.strides[0]/sizeof(T);
	int Hs = binfo.strides[1]/sizeof(T);

	for(int b=0; b<B; b++) {
		for(int h=0; h<H; h++) {
			_to[b*H+h] = _from[b*Bs+h*Hs];
		}
	}
}

template<typename T> void Buffer::copy3(void* to, py::buffer_info &binfo)
{
	T* _to = (T*) to;
	T* _from = (T*) binfo.ptr;
	int B = binfo.shape[0];
	int H = binfo.shape[1];
	int W = binfo.shape[2];

	int Bs = binfo.strides[0]/sizeof(T);
	int Hs = binfo.strides[1]/sizeof(T);
	int Ws = binfo.strides[2]/sizeof(T);

	for(int b=0; b<B; b++) {
		for(int h=0; h<H; h++) {
			for(int w=0; w<W; w++) {
				_to[(b*H+h)*W+w] = _from[b*Bs+h*Hs+w*Ws];
			}
		}
	}
}

template<typename T> void Buffer::copy4(void* to, py::buffer_info &binfo)
{
	T* _to = (T*) to;
	T* _from = (T*) binfo.ptr;
	int B = binfo.shape[0];
	int H = binfo.shape[1];
	int W = binfo.shape[2];
	int C = binfo.shape[3];

	int Bs = binfo.strides[0]/sizeof(T);
	int Hs = binfo.strides[1]/sizeof(T);
	int Ws = binfo.strides[2]/sizeof(T);
	int Cs = binfo.strides[3]/sizeof(T);

	for(int b=0; b<B; b++) {
		for(int h=0; h<H; h++) {
			for(int w=0; w<W; w++) {
				for(int c=0; c<C; c++) {
					_to[((b*H+h)*W+w)*C+c] = _from[b*Bs+h*Hs+w*Ws+c*Cs];
				}
			}
		}
	}
}

template<typename T> void Buffer::copy(void* to, py::buffer_info &binfo)
{

	switch(m_NDim) {
		case 1:
			memcpy(to, binfo.ptr, m_Size*m_ItemSize);
			break;
		case 2:
			copy2<T>(to, binfo);
			break;
		case 3:
			copy3<T>(to, binfo);
			break;
		case 4:
			copy4<T>(to, binfo);
			break;
		default:
			throw std::runtime_error("invalid dimensions");
			break;
	}
}

void Buffer::load(py::array& array)
{
	void* to;
	py::buffer_info binfo = array.request();

	if(m_ItemSize != m_ArrayItemSize) {
		to = m_ArrayData;
	} else {
		to = m_Data;
	}

	switch(m_ArrayItemSize) {
		case 1: copy<uint8_t>(to, binfo); break;
		case 2: copy<uint16_t>(to, binfo); break;
		case 4: copy<uint32_t>(to, binfo); break;
		default:
			throw std::runtime_error("invalid array item size");
			break;
	}
}

void Buffer::reload(py::array& array)
{
	py::buffer_info binfo = array.request();

	validate(binfo);

	if(m_DataMalloced) {
		if(m_Size < binfo.size) {
			m_Size = binfo.size;
			free(m_Data);
			if(m_ArrayData) free(m_ArrayData);
			m_Data = malloc(m_Size*m_ItemSize);
			if(nullptr == m_Data) {
				throw std::runtime_error("No memory for m_Data");
			}
			if(m_ItemSize != m_ArrayItemSize) {
				m_ArrayData = malloc(m_Size*m_ArrayItemSize);
				if(nullptr == m_ArrayData) {
					throw std::runtime_error("No memory for m_ArrayData");
				}
			}
		}
	}

	load(array);
}

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

	for(std::pair<const layer_t*, Buffer*> item: m_Buffers) {
		Buffer* buffer = item.second;
		delete buffer;
	}
}

void Model::populate_inputs(py::dict& feed)
{
	void* data = nullptr;
	const nn_input_t* const* input = nn->network->inputs;

	while((*input) != nullptr)
	{
		const layer_t* layer = (*input)->layer;
		data = (*input)->data;
		py::array array = feed[layer->name];
		Buffer* buffer;
		if(m_Buffers.find(layer) == m_Buffers.end()) {
			buffer = new Buffer(data, layer, array);
			m_Buffers.insert(std::pair<const layer_t*, Buffer*>(layer, buffer));
		} else {
			buffer = m_Buffers[layer];
			buffer->reload(array);
		}

		input++;
	}
}

py::object Model::predict(py::dict feed)
{
	populate_inputs(feed);

	int r = nn_predict(nn);

	if(r != 0) {
		throw std::runtime_error(string_format("Failed to predict, error is %d", r));
	}
	return py::none();
}
}
