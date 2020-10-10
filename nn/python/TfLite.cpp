/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020 Parai Wang <parai@foxmail.com>
 */
#ifdef ENABLE_TFLITE
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
TfBuffer::TfBuffer(TfLiteTensor* tensor, py::array& array)
		: m_Tensor(tensor), m_Data(tensor->data.data)
{
	py::buffer_info binfo = array.request();

	m_Size = binfo.size;
	m_ArrayItemSize = binfo.itemsize;
	switch(tensor->type) {
		case kTfLiteUInt8: case kTfLiteInt8: m_ItemSize = 1; break;
		case kTfLiteInt16: m_ItemSize = 2; break;
		case kTfLiteFloat32: m_ItemSize = 4; break;
		default: throw std::runtime_error("invalid tensor data type"); break;
	}
	m_LayerSize = 1;
	for(int i=0; i<m_Tensor->dims->size; i++) {
		m_LayerSize *= m_Tensor->dims->data[i];
	}
	m_NDim = tensor->dims->size;
	validate(binfo);

	if(nullptr == m_Data) {
		throw std::runtime_error("No memory for m_Data");
	}

	if(m_ItemSize != m_ArrayItemSize) {
		m_ArrayData = malloc(m_Size*m_ArrayItemSize);
		if(nullptr == m_ArrayData) {
			throw std::runtime_error("No memory for m_ArrayData");
		}
	}

	load(array);
};

TfBuffer::TfBuffer(TfLiteTensor* tensor)
		: m_Tensor(tensor), m_Data(tensor->data.data)
{
	m_NDim = tensor->dims->size;
	m_Size = 1;
	for(int i=0; i<m_Tensor->dims->size; i++) {
		m_Size *= m_Tensor->dims->data[i];
	}
	if(nullptr == m_Data) {
		throw std::runtime_error("No memory for m_Data");
	}

	switch(tensor->type) {
		case kTfLiteUInt8: case kTfLiteInt8: m_ItemSize = 1; break;
		case kTfLiteInt16: m_ItemSize = 2; break;
		case kTfLiteFloat32: m_ItemSize = 4; break;
		default: throw std::runtime_error("invalid tensor data type"); break;
	}
}

TfBuffer::TfBuffer(py::array& array)
{
	py::buffer_info binfo = array.request();

	m_Size = binfo.size;
	m_ItemSize = binfo.itemsize;
	m_NDim = binfo.ndim;
	m_Data = malloc(m_Size*m_ItemSize);
	m_DataMalloced = true;
	if(nullptr == m_Data) {
		throw std::runtime_error("No memory for m_Data");
	}
	switch(m_ItemSize) {
		case 4:
			copy<float>(m_Data, binfo);
			break;
		default:
			throw std::runtime_error("invalid item size");
			break;
	}

}

TfBuffer::~TfBuffer()
{
	if(m_DataMalloced && (nullptr != m_Data)){
		free(m_Data);
	}

	if(nullptr != m_ArrayData) {
		free(m_ArrayData);
	}
};

py::array TfBuffer::numpy()
{
	py::array array;
	m_LayerSize = 1;
	m_NDim = m_Tensor->dims->size;
	for(int i=0; i<m_Tensor->dims->size; i++) {
		m_LayerSize *= m_Tensor->dims->data[i];
	}
	if(m_DataMalloced) {
		if(m_Size < m_LayerSize) {
			if(nullptr != m_Data) free(m_Data);
			m_Size = m_LayerSize;
			m_Data = malloc(m_Size*m_ItemSize);
			if(nullptr == m_Data) {
				throw std::runtime_error("No memory for m_Data");
			}
		}

		memcpy(m_Data, m_Tensor->data.data, m_Size*m_ItemSize);
	}

	switch(m_NDim) {
		case 1:
			array = py::array_t<float>({(size_t)m_Tensor->dims->data[0]});
			break;
		case 2:
			array = py::array_t<float>({(size_t)m_Tensor->dims->data[0], (size_t)m_Tensor->dims->data[1]});
			break;
		case 3:
			array = py::array_t<float>({(size_t)m_Tensor->dims->data[0], (size_t)m_Tensor->dims->data[1], (size_t)m_Tensor->dims->data[2]});
			break;
		case 4:
			array = py::array_t<float>({(size_t)m_Tensor->dims->data[0], (size_t)m_Tensor->dims->data[1], (size_t)m_Tensor->dims->data[2], (size_t)m_Tensor->dims->data[3]});
			break;
		default:
			throw std::runtime_error("invalid dimension");
			break;
	}

	py::buffer_info binfo = array.request();

	if(binfo.itemsize == m_ItemSize) {
		memcpy(binfo.ptr, m_Data, m_LayerSize*m_ItemSize);
	} else {
		switch(m_Tensor->type) {
			case kTfLiteUInt8:
				dequantize<uint8_t>((float*)binfo.ptr, (uint8_t*)m_Data);
				break;
			case kTfLiteInt8:
				dequantize<int8_t>((float*)binfo.ptr, (int8_t*)m_Data);
				break;
			default: throw std::runtime_error("invalid tensor data type"); break;
		}
	}

	return array;
}

void TfBuffer::validate(py::buffer_info &binfo)
{
	bool valid = true;

	if(m_ArrayItemSize != binfo.itemsize) {
		throw std::runtime_error("invalid data type");
	}

	if(m_ArrayItemSize != m_ItemSize) {
		if( m_ItemSize != 1 ) {
			throw std::runtime_error("only support Q8");
		}

		if(m_ArrayItemSize != 4) {
			throw std::runtime_error(string_format("need float input for Q%d", m_ItemSize*8));
		}
	}

	if((size_t)m_LayerSize != m_Size) {
		valid = false;
	} else if(m_NDim != binfo.ndim) {
		valid = false;
	} else {
		for(int i=0; i<m_NDim; i++) {
			if(m_Tensor->dims->data[i] != binfo.shape[i]) {
				valid = false;
			}
		}
	}

	if(false == valid) {
		char msg[256];
		int len = snprintf(msg, sizeof(msg), "Invalid TfBuffer: required %d (", m_LayerSize);
		for(int i=0; i<m_NDim; i++) {
			len += snprintf(&msg[len], sizeof(msg)-len, "%d,", m_Tensor->dims->data[i]);
		}
		len += snprintf(&msg[len], sizeof(msg)-len, "), provided %d (", (int)m_Size);
		for(int i=0; i<binfo.ndim; i++) {
			len += snprintf(&msg[len], sizeof(msg)-len, "%d,", (int)binfo.shape[i]);
		}
		len += snprintf(&msg[len], sizeof(msg)-len, ")");
		throw std::runtime_error(msg);
	}
}

template<typename T> void TfBuffer::copy2(void* to, py::buffer_info &binfo)
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

template<typename T> void TfBuffer::copy3(void* to, py::buffer_info &binfo)
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

template<typename T> void TfBuffer::copy4(void* to, py::buffer_info &binfo)
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

template<typename T> void TfBuffer::copy(void* to, py::buffer_info &binfo)
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

void TfBuffer::load(py::array& array)
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

	if(need_quantize()) {
		switch(m_Tensor->type) {
			case kTfLiteUInt8:
				quantize<uint8_t>((uint8_t*)m_Data, (float*)m_ArrayData);
				break;
			case kTfLiteInt8:
				quantize<int8_t>((int8_t*)m_Data, (float*)m_ArrayData);
				break;
			default: throw std::runtime_error("invalid tensor data type"); break;
		}
	}
}

void TfBuffer::reload(py::array& array)
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

template<typename T> void TfBuffer::quantize(T* out, float* in)
{
	float scale = m_Tensor->params.scale;
	int32_t zero = m_Tensor->params.zero_point;

	for(size_t i=0; i < m_Size; i++) {
		out[i] = (T)std::round(in[i]/scale + zero);
	}
}

template<typename T> void TfBuffer::dequantize(float* out, T* in)
{
	float scale = m_Tensor->params.scale;
	int32_t zero = m_Tensor->params.zero_point;

	for(size_t i=0; i < m_Size; i++) {
		out[i] = scale*(in[i] - zero);
	}
}

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

	if (m_Model->version() != TFLITE_SCHEMA_VERSION) {
		throw std::runtime_error(string_format(
				"Model provided is schema version %d not equal "
				"to supported version %d.",
				m_Model->version(), TFLITE_SCHEMA_VERSION));
	}

	m_TensorArea = new uint8_t[kTensorArenaSize];
	m_Interpreter = new tflite::MicroInterpreter(
			m_Model, resolver, m_TensorArea, kTensorArenaSize, &error_reporter);

	TfLiteStatus allocate_status = m_Interpreter->AllocateTensors();
	if (allocate_status != kTfLiteOk) {
		throw std::runtime_error("AllocateTensors() failed");
	}
}
TfLite::~TfLite()
{
	delete m_Interpreter;
	delete[] m_TensorArea;
	delete[] m_ModelData;
	for(std::pair<TfLiteTensor*, TfBuffer*> item: m_Buffers) {
		TfBuffer* TfBuffer = item.second;
		delete TfBuffer;
	}
}

void TfLite::populate_inputs(py::dict& feed)
{
	NNLOG(NN_DEBUG, ("%d Inputs:\n", (int)m_Interpreter->inputs_size()));
	for(int i = 0; i < m_Interpreter->inputs_size(); i++) {
		TfLiteTensor* tensor = m_Interpreter->input_tensor(i);
		NNLOG(NN_DEBUG, (" [%d]: Type %d, Dims:[", i, (int)tensor->type));
		for(int d = 0; d < tensor->dims->size; d++) {
			NNLOG(NN_DEBUG, (" %d", tensor->dims->data[d]));
		}
		NNLOG(NN_DEBUG, (" ], "));
		switch(tensor->type) {
			case kTfLiteInt32:
			case kTfLiteUInt8:
			case kTfLiteInt16:
			case kTfLiteInt8:
				NNLOG(NN_DEBUG, ("Quantization: scale=%f, zero=%d",
						tensor->params.scale, tensor->params.zero_point));
				break;
			default:
				break;
		}
		NNLOG(NN_DEBUG, ("\n"));

		py::array array = feed[string_format("%d", i).c_str()];
		TfBuffer* buffer;
		if(m_Buffers.find(tensor) == m_Buffers.end()) {
			buffer = new TfBuffer(tensor, array);
			m_Buffers.insert(std::pair<TfLiteTensor*, TfBuffer*>(tensor, buffer));
		} else {
			buffer = m_Buffers[tensor];
			buffer->reload(array);
		}
	}
}

void TfLite::populate_outputs(py::dict& outputs)
{
	NNLOG(NN_DEBUG, ("%d Outputs:\n", (int)m_Interpreter->outputs_size()));
	for(int i = 0; i < m_Interpreter->outputs_size(); i++) {
		TfLiteTensor* tensor = m_Interpreter->output_tensor(i);
		NNLOG(NN_DEBUG, (" [%d]: Type %d, Dims:[", i, (int)tensor->type));
		for(int d = 0; d < tensor->dims->size; d++) {
			NNLOG(NN_DEBUG, (" %d", tensor->dims->data[d]));
		}
		NNLOG(NN_DEBUG, (" ], "));
		switch(tensor->type) {
			case kTfLiteInt32:
			case kTfLiteUInt8:
			case kTfLiteInt16:
			case kTfLiteInt8:
				NNLOG(NN_DEBUG, ("Quantization: scale=%f, zero=%d",
						tensor->params.scale, tensor->params.zero_point));
				break;
			default:
				break;
		}
		NNLOG(NN_DEBUG, ("\n"));

		TfBuffer* buffer;
		if(m_Buffers.find(tensor) == m_Buffers.end()) {
			buffer = new TfBuffer(tensor);
			m_Buffers.insert(std::pair<TfLiteTensor*, TfBuffer*>(tensor, buffer));
		} else {
			buffer = m_Buffers[tensor];
		}
		outputs[string_format("%d", i).c_str()] = buffer->numpy();
	}
}

py::object TfLite::predict(py::dict feed)
{
	py::dict outputs;

	populate_inputs(feed);

	TfLiteStatus invoke_status = m_Interpreter->Invoke();
	if (invoke_status != kTfLiteOk) {
		throw std::runtime_error("Invoke failed\n");
	}

	populate_outputs(outputs);

	return outputs;
}
}
#endif
