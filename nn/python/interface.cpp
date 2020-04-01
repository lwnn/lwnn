/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "nn.h"
#include "algorithm.h"
#include <math.h>
#include <cmath>
#include <vector>
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
int alg_up_sampling(void* pout, void* pin, NHWC_t *outNHWC, NHWC_t *inNHWC, size_t type_size, uint8_t* pmask);
int ROIAlign_forward_cpu(float* o, const float* in, const float* boxes, const int* indices,
		NHWC_t* onhwc, NHWC_t* inhwc);
int CropAndResize_forward_cpu(float* o, const float* in, const float* boxes, const int* indices,
		NHWC_t* onhwc, NHWC_t* inhwc);
}
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
namespace py = pybind11;

void set_log_level(int level)
{
	nn_set_log_level(level);
}

py::object MaxPool2d(py::array_t<float> input, /* NCHW */
		py::array_t<int> kernel_size,
		py::array_t<int> stride,
		py::array_t<int> padding,
		py::array_t<int> output_shape, /* NCHW */
		bool with_mask) {
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

	auto result_data = py::array_t<float>({
				(size_t)batches, (size_t)ch_im_out,
				(size_t)dim_im_out_y, (size_t)dim_im_out_x});

	py::buffer_info buf_out = result_data.request();

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
	uint8_t * mask = NULL;

	if(with_mask) {
		mask = new uint8_t[batches*batch_sizeO];
	}

	NHWC_t inhwc = { batches, dim_im_in_y, dim_im_in_x, ch_im_in };
	alg_transpose(in, IN, &inhwc, sizeof(float), ALG_TRANSPOSE_FROM_NCHW_TO_NHWC);

	NNDDO(NN_DEBUG, rte_save_raw("maxpool_in.raw", in, batches*batch_sizeIn*sizeof(float)));

	for(int batch=0; batch<batches; batch++)
	{
		uint8_t *M = NULL;
		if(with_mask) {
			M = mask+batch_sizeO*batch;
		}
		pooling(in+batch_sizeIn*batch,
				dim_im_in_x, dim_im_in_y,
				ch_im_in, ch_im_out,
				dim_kernel_x, dim_kernel_y,
				padding_x, padding_y,
				stride_x, stride_y,
				o+batch_sizeO*batch,
				dim_im_out_x,dim_im_out_y,
				L_OP_MAXPOOL,
				M);
	}

	NHWC_t onhwc = { batches, dim_im_out_y, dim_im_out_x, ch_im_out };
	alg_transpose(O, o, &onhwc, sizeof(float), ALG_TRANSPOSE_FROM_NHWC_TO_NCHW);

	NNDDO(NN_DEBUG, rte_save_raw("maxpool_out.raw", O, batches*batch_sizeO*sizeof(float)));

	delete in;
	delete o;
	if(with_mask) {
		auto result_mask = py::array_t<uint8_t>({
					(size_t)batches, (size_t)ch_im_out,
					(size_t)dim_im_out_y, (size_t)dim_im_out_x});

		py::buffer_info buf_mask = result_mask.request();
		uint8_t *O_mask = (uint8_t *) buf_mask.ptr;
		alg_transpose(O_mask, mask, &onhwc, sizeof(uint8_t), ALG_TRANSPOSE_FROM_NHWC_TO_NCHW);
		py::list result;
		result.append(result_data);
		result.append(result_mask);
		delete mask;
		return result;
	}

	return result_data;
}

py::object Upsample2d(py::array_t<float> input, /* NCHW */
		py::array_t<int> output_shape, /* NCHW */
		py::array_t<uint8_t> mask) {
	py::buffer_info buf_in = input.request();

	if (buf_in.ndim != 4)
		throw std::runtime_error("Number of dimensions must be 4");

	int N = buf_in.shape[0];
	int iH = buf_in.shape[2];
	int iW = buf_in.shape[3];
	int iC = buf_in.shape[1];

	int oH = (int)output_shape.at(2);
	int oW = (int)output_shape.at(3);
	int oC = (int)output_shape.at(1);

	auto result = py::array_t<float>({
				(size_t)N, (size_t)oC,
				(size_t)oH, (size_t)oW});

	py::buffer_info buf_out = result.request();

	float *IN = (float *) buf_in.ptr;
	float *O = (float *) buf_out.ptr;
	uint8_t *Mask_out = NULL;

	float * in = new float[N*iH*iW*iC];
	float * o  = new float[N*oH*oW*oC];
	uint8_t * pmask = NULL;

	NNLOG(NN_DEBUG, ("Upsample2d%s:", mask.is_none()?"":" with mask"));

	NHWC_t inhwc = { N, iH, iW, iC };

	if(!mask.is_none()) {
		py::buffer_info buf_mask = mask.request();
		Mask_out = (uint8_t *) buf_mask.ptr;
		pmask = new uint8_t[N*iH*iW*iC];
		alg_transpose(pmask, Mask_out, &inhwc, sizeof(uint8_t), ALG_TRANSPOSE_FROM_NCHW_TO_NHWC);
	}

	alg_transpose(in, IN, &inhwc, sizeof(float), ALG_TRANSPOSE_FROM_NCHW_TO_NHWC);

	NHWC_t onhwc = { N, oH, oW, oC };
	alg_up_sampling(o, in, &onhwc, &inhwc, sizeof(float), pmask);
	alg_transpose(O, o, &onhwc, sizeof(float), ALG_TRANSPOSE_FROM_NHWC_TO_NCHW);

	delete in;
	delete o;
	if(pmask) {
		delete pmask;
	}

	return result;
}

py::array_t<float> PriorBox(py::array_t<int> feature_shape, /* NCHW */
		py::array_t<int> image_shape,
		py::array_t<float> variance,
		py::array_t<int> max_sizes,
		py::array_t<int> min_sizes,
		py::array_t<float> aspect_ratios,
		int clip, int flip,
		float step, float offset,
		py::array_t<int> output_shape) {

	auto result = py::array_t<float>({ (size_t)output_shape.at(0),
									   (size_t)output_shape.at(1),
									   (size_t)output_shape.at(2)});
	py::buffer_info buf_out = result.request();
	float *top_data = (float *) buf_out.ptr;

	int layer_width = (int) feature_shape.at(3);
	int layer_height = (int) feature_shape.at(2);

	int img_width = (int) image_shape.at(3);
	int img_height = (int) image_shape.at(2);

	float step_w = step;
	float step_h = step;

	if (0 == step) {
		step_w = static_cast<float>(img_width) / layer_width;
		step_h = static_cast<float>(img_height) / layer_height;
	}

	int num_priors_ = (int) output_shape.at(2) / (4 * layer_width * layer_height);

	int num_ar = aspect_ratios.request().shape[0];
	int num_max = max_sizes.request().shape[0];
	int num_min = min_sizes.request().shape[0];
	int num_var = variance.request().shape[0];

	NNLOG(NN_DEBUG, ("PriorBox: layer.(w,h)=(%d,%d) img.(w,h)=(%d,%d), step.(w,h)=(%.2f,%.2f), offset=%.2f, num_priors=%d, num_ar=%d\n",
			layer_width, layer_height, img_width, img_height, step_w, step_h, offset, num_priors_, num_ar));

	std::vector<float> aspect_ratios_;
	aspect_ratios_.push_back(1.);
	for (int i = 0; i < num_ar; ++i) {
		float ar = (float)aspect_ratios.at(i);
		bool already_exist = false;
		for (int j = 0; j < aspect_ratios_.size(); ++j) {
			if (fabs(ar - aspect_ratios_[j]) < 1e-6) {
				already_exist = true;
				break;
			}
		}
		if (!already_exist) {
			aspect_ratios_.push_back(ar);
			if (flip) {
				aspect_ratios_.push_back(1. / ar);
			}
		}
	}

	int dim = layer_height * layer_width * num_priors_ * 4;
	assert(dim == (int) output_shape.at(2));
	int idx = 0;
	for (int h = 0; h < layer_height; ++h) {
		for (int w = 0; w < layer_width; ++w) {
			float center_x = (w + offset) * step_w;
			float center_y = (h + offset) * step_h;
			float box_width, box_height;
			for (int s = 0; s < num_min; ++s) {
				int min_size_ = (int) min_sizes.at(s);
				// first prior: aspect_ratio = 1, size = min_size
				box_width = box_height = min_size_;
				// xmin
				top_data[idx++] = (center_x - box_width / 2.) / img_width;
				// ymin
				top_data[idx++] = (center_y - box_height / 2.) / img_height;
				// xmax
				top_data[idx++] = (center_x + box_width / 2.) / img_width;
				// ymax
				top_data[idx++] = (center_y + box_height / 2.) / img_height;

				if (num_max > 0) {
					assert(num_min == num_max);
					int max_size_ = (int) max_sizes.at(s);
					// second prior: aspect_ratio = 1, size = sqrt(min_size * max_size)
					box_width = box_height = sqrt(min_size_ * max_size_);
					// xmin
					top_data[idx++] = (center_x - box_width / 2.) / img_width;
					// ymin
					top_data[idx++] = (center_y - box_height / 2.) / img_height;
					// xmax
					top_data[idx++] = (center_x + box_width / 2.) / img_width;
					// ymax
					top_data[idx++] = (center_y + box_height / 2.) / img_height;
				}

				// rest of priors
				for (int r = 0; r < aspect_ratios_.size(); ++r) {
					float ar = aspect_ratios_[r];
					if (fabs(ar - 1.) < 1e-6) {
						continue;
					}
					box_width = min_size_ * sqrt(ar);
					box_height = min_size_ / sqrt(ar);
					// xmin
					top_data[idx++] = (center_x - box_width / 2.) / img_width;
					// ymin
					top_data[idx++] = (center_y - box_height / 2.) / img_height;
					// xmax
					top_data[idx++] = (center_x + box_width / 2.) / img_width;
					// ymax
					top_data[idx++] = (center_y + box_height / 2.) / img_height;
				}
			}
		}
	}
	// clip the prior's coordidate such that it is within [0, 1]
	if (clip) {
		for (int d = 0; d < dim; ++d) {
			top_data[d] = std::min<float>(std::max<float>(top_data[d], 0.), 1.);
		}
	}

	// set the variance.
	top_data += dim;
	if (num_var == 1) {
		for (int i = 0; i < dim; i++) {
			top_data[i] = (float) variance.at(0);
		}
	} else {
		int count = 0;
		for (int h = 0; h < layer_height; ++h) {
			for (int w = 0; w < layer_width; ++w) {
				for (int i = 0; i < num_priors_; ++i) {
					for (int j = 0; j < 4; ++j) {
						top_data[count] = variance.at(j);
						++count;
					}
				}
			}
		}
	}

	return result;
}

py::array_t<float> ROIAlign(py::array_t<float> X, /* NCHW */
		py::array_t<float> rois, /* (num_rois,4) */
		py::array_t<int> batch_indices, /* (num_rois) */
		int output_height, int output_width, int mode=0) {
	py::buffer_info buf_in = X.request();
	py::buffer_info buf_rois = rois.request();
	py::buffer_info buf_batch_ind = batch_indices.request();

	if (buf_in.ndim != 4)
		throw std::runtime_error("Number of dimensions of X must be 4");
	if (buf_rois.ndim != 2)
		throw std::runtime_error("Number of dimensions of rois must be 2");
	if (buf_batch_ind.ndim != 1)
		throw std::runtime_error("Number of dimensions of batch_indices must be 1");
	if (buf_rois.shape[0] != buf_batch_ind.shape[0])
		throw std::runtime_error("shape[0] of rois and batch_indices must be equal");

	int iN = buf_in.shape[0];
	int iH = buf_in.shape[2];
	int iW = buf_in.shape[3];
	int iC = buf_in.shape[1];

	int oN = buf_rois.shape[0];
	int oH = output_height;
	int oW = output_width;
	int oC = iC;

	auto result = py::array_t<float>({(size_t)oN, (size_t)oC, (size_t)oH, (size_t)oW});

	py::buffer_info buf_out = result.request();

	float *IN = (float *) buf_in.ptr;
	float *O = (float *) buf_out.ptr;
	float *boxes = (float *) buf_rois.ptr;
	int* indices = (int *) buf_batch_ind.ptr;

	float * in = new float[iN*iH*iW*iC];
	float * o  = new float[oN*oH*oW*oC];
	uint8_t * pmask = NULL;

	NHWC_t inhwc = { iN, iH, iW, iC };

	alg_transpose(in, IN, &inhwc, sizeof(float), ALG_TRANSPOSE_FROM_NCHW_TO_NHWC);

	NHWC_t onhwc = { oN, oH, oW, oC };
	if(mode == 0) {
		ROIAlign_forward_cpu(o, in, boxes, indices, &onhwc, &inhwc);
	} else {
		CropAndResize_forward_cpu(o, in, boxes, indices, &onhwc, &inhwc);
	}
	alg_transpose(O, o, &onhwc, sizeof(float), ALG_TRANSPOSE_FROM_NHWC_TO_NCHW);

	delete in;
	delete o;

	return result;
}
PYBIND11_MODULE(liblwnn, m)
{
	m.doc() = "pybind11 lwnn plugin";
	m.def("set_log_level", &set_log_level, "set lwnn log level");
	m.def("MaxPool2d", &MaxPool2d, "lwnn functional MaxPool2d",
			py::arg("input"), py::arg("kernel_size"), py::arg("stride"),
			py::arg("padding"), py::arg("output_shape"),
			py::arg("with_mask")=false);
	m.def("Upsample2d", &Upsample2d, "lwnn functional Upsample2d",
			py::arg("input"), py::arg("output_shape"), py::arg("mask")=py::none());
	m.def("PriorBox", &PriorBox, "lwnn functional PriorBox",
			py::arg("feature_shape"), py::arg("image_shape"), py::arg("variance"),
			py::arg("max_sizes"), py::arg("min_sizes"), py::arg("aspect_ratio"),
			py::arg("clip"), py::arg("flip"), py::arg("step"), py::arg("offset"),
			py::arg("output_shape"));
	m.def("ROIAlign", &ROIAlign, "lwnn functional ROTAlign",
			py::arg("X"), py::arg("rois"), py::arg("batch_indices"),
			py::arg("output_height")=1, py::arg("output_width")=1, py::arg("mode")=0);
}


