/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nn.h"
/* https://pybind11.readthedocs.io/en/stable/advanced/pycpp/numpy.html */
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
extern "C" {
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

py::array_t<float> MaxPool2d(py::array_t<float> input, py::array_t<int> kernel_size,
		py::array_t<int> stride, py::array_t<int> padding, py::array_t<int> output_shape) {
	py::buffer_info buf_in = input.request();

	if (buf_in.ndim != 4)
		throw std::runtime_error("Number of dimensions must be 4");

	auto batches = buf_in.shape[0];
	int dim_im_in_y = buf_in.shape[1];
	int dim_im_in_x = buf_in.shape[2];
	int ch_im_in = buf_in.shape[3];
	size_t batch_sizeIn =ch_im_in*dim_im_in_y*dim_im_in_x;

	int dim_im_out_y = (int)output_shape.at(1);
	int dim_im_out_x = (int)output_shape.at(2);
	int ch_im_out = (int)output_shape.at(3);
	size_t batch_sizeO =ch_im_out*dim_im_out_y*dim_im_out_x;

	int dim_kernel_y = (int)kernel_size.at(0);
	int dim_kernel_x = (int)kernel_size.at(1);

	int padding_y = (int)padding.at(0);
	int padding_x = (int)padding.at(1);

	int stride_y = (int)stride.at(0);
	int stride_x = (int)stride.at(1);

	/* No pointer is passed, so NumPy will allocate the buffer */
	auto result = py::array_t<float>({batches, dim_im_out_y, dim_im_out_x, ch_im_out});

	py::buffer_info buf_out = result.request();

	float *IN = (float *) buf_in.ptr;
	float *O = (float *) buf_out.ptr;

	for(int batch=0; batch<batches; batch++)
	{
		pooling(IN+batch_sizeIn*batch,
				dim_im_in_x, dim_im_in_y,
				ch_im_in, ch_im_out,
				dim_kernel_x, dim_kernel_y,
				padding_x, padding_y,
				stride_x, stride_y,
				O+batch_sizeO*batch,
				dim_im_out_x,dim_im_out_y,
				L_OP_MAXPOOL,
				NULL);
	}

	return result;
}

PYBIND11_PLUGIN(liblwnn)
{
	py::module m("liblwnn", "pybind11 lwnn plugin");
	m.def("MaxPool2d", &MaxPool2d, "lwnn functional MaxPool2d",
			py::arg("input"), py::arg("kernel_size"), py::arg("stride"),
			py::arg("padding"), py::arg("output_shape"));
	return m.ptr();
}
