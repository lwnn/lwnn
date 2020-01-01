/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nn.h"
#include "algorithm.h"
/* https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html */
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
extern "C" {
int rte_load_raw(const char* name, void* data, size_t sz);
void rte_save_raw(const char* name, void* data, size_t sz);
int pooling(const float * Im_in,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const int ch_im_out,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		float * Im_out,
		const int dim_im_out_x,
		const int dim_im_out_y,
		layer_operation_t op,
		uint8_t* Mask_out);
}
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
namespace py = pybind11;

void set_log_level(int level)
{
	nn_set_log_level(level);
}

py::array_t<float> MaxPool2d(py::array_t<float> input, /* NCHW */
		py::array_t<int> kernel_size,
		py::array_t<int> stride,
		py::array_t<int> padding,
		py::array_t<int> output_shape /* NCHW */) {
	py::buffer_info buf_in = input.request();

	if (buf_in.ndim != 4)
		throw std::runtime_error("Number of dimensions must be 4");

	int batches = buf_in.shape[0];
	int dim_im_in_y = buf_in.shape[2];
	int dim_im_in_x = buf_in.shape[3];
	int ch_im_in = buf_in.shape[1];
	size_t batch_sizeIn =ch_im_in*dim_im_in_y*dim_im_in_x;

	NNLOG(NN_DEBUG, ("MaxPool2d: in shape = [%d %d %d %d],", batches, dim_im_in_y, dim_im_in_x, ch_im_in));

	int dim_im_out_y = (int)output_shape.at(2);
	int dim_im_out_x = (int)output_shape.at(3);
	int ch_im_out = (int)output_shape.at(1);
	size_t batch_sizeO =ch_im_out*dim_im_out_y*dim_im_out_x;

	NNLOG(NN_DEBUG, (" out shape = [%d %d %d],", dim_im_out_y, dim_im_out_x, ch_im_out));

	int dim_kernel_y = (int)kernel_size.at(0);
	int dim_kernel_x = (int)kernel_size.at(1);

	NNLOG(NN_DEBUG, (" kernel = [%d %d],", dim_kernel_y, dim_kernel_x));

	int padding_y = (int)padding.at(0);
	int padding_x = (int)padding.at(1);

	NNLOG(NN_DEBUG, (" padding = [%d %d],", padding_y, padding_x));

	int stride_y = (int)stride.at(0);
	int stride_x = (int)stride.at(1);

	NNLOG(NN_DEBUG, (" stride = [%d %d]\n", stride_y, stride_x));

	auto result = py::array_t<float>({
				(size_t)batches, (size_t)ch_im_out,
				(size_t)dim_im_out_y, (size_t)dim_im_out_x});

	py::buffer_info buf_out = result.request();

	NNLOG(NN_DEBUG, ("  buf_in stride(%d) = [%d %d %d %d],", (int)buf_in.strides.size(),
			(int)buf_in.strides[0], (int)buf_in.strides[1],
			(int)buf_in.strides[2], (int)buf_in.strides[3]));
	NNLOG(NN_DEBUG, ("  buf_out stride(%d) = [%d %d %d %d]\n", (int)buf_out.strides.size(),
			(int)buf_out.strides[0], (int)buf_out.strides[1],
			(int)buf_out.strides[2], (int)buf_out.strides[3]));

	float *IN = (float *) buf_in.ptr;
	float *O = (float *) buf_out.ptr;

	float * in = new float[batches*batch_sizeIn];
	float * o  = new float[batches*batch_sizeO];

	NHWC_t inhwc = { batches, dim_im_in_y, dim_im_in_x, ch_im_in };
	alg_transpose(in, IN, &inhwc, sizeof(float), ALG_TRANSPOSE_FROM_NCHW_TO_NHWC);

	NNDDO(NN_DEBUG, rte_save_raw("maxpool_in.raw", in, batches*batch_sizeIn*sizeof(float)));

	for(int batch=0; batch<batches; batch++)
	{
		pooling(in+batch_sizeIn*batch,
				dim_im_in_x, dim_im_in_y,
				ch_im_in, ch_im_out,
				dim_kernel_x, dim_kernel_y,
				padding_x, padding_y,
				stride_x, stride_y,
				o+batch_sizeO*batch,
				dim_im_out_x,dim_im_out_y,
				L_OP_MAXPOOL,
				NULL);
	}

	NHWC_t onhwc = { batches, dim_im_out_y, dim_im_out_x, ch_im_out };
	alg_transpose(O, o, &onhwc, sizeof(float), ALG_TRANSPOSE_FROM_NHWC_TO_NCHW);

	NNDDO(NN_DEBUG, rte_save_raw("maxpool_out.raw", O, batches*batch_sizeO*sizeof(float)));

	delete in;
	delete o;

	return result;
}

PYBIND11_PLUGIN(liblwnn)
{
	py::module m("liblwnn", "pybind11 lwnn plugin");
	m.def("set_log_level", &set_log_level, "set lwnn log level");
	m.def("MaxPool2d", &MaxPool2d, "lwnn functional MaxPool2d",
			py::arg("input"), py::arg("kernel_size"), py::arg("stride"),
			py::arg("padding"), py::arg("output_shape"));
	return m.ptr();
}
