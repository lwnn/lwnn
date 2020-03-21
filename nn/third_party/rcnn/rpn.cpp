/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
/* ============================ [ MACROS    ] ====================================================== */
namespace py = pybind11;
using namespace py::literals;
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
extern "C" void rte_save_raw(const char* name, void* data, size_t sz);
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
template<typename T>
static py::array create_1d_array(const layer_blob_t* blob) {
    size_t ndim = blob->dims[0];
    auto arr = py::array_t<T>({ndim});
    py::buffer_info arr_info = arr.request();
    T* arr_data = (T*) arr_info.ptr;
    T* data = (T*)blob->blob;
    std::copy(data, data+ndim, arr_data);
    return arr;
}

static py::object create_array(const layer_blob_t* blob) {
    int dim = 0;
    const int* dims = blob->dims;

    assert(dims != NULL);

    while((dims[dim] != 0) && (dim < 4)) {
        dim ++;
    };

    if(1 == dim) {
        switch(blob->dtype) {
        case L_DT_INT32:
            return create_1d_array<int32_t>(blob);
            break;
        case L_DT_FLOAT:
            return create_1d_array<float>(blob);
            break;
        default:
            break;
        }
    }

    printf("unsupported blob with dim=%d, dtype=%d\n", dim, blob->dtype);
    assert(0);

    return py::none();
}
/* ============================ [ FUNCTIONS ] ====================================================== */
extern "C" int rpn_generate_anchors(const nn_t* nn, const layer_t* layer, float** anchors)
{
  int r = 0;
  if(NULL == getenv("PYTHONHOME")) {
    printf("WARNING: env PYTHONHOME is not set, please set it correctly if see Py_Initialize error\n");
  }
  py::scoped_interpreter guard{};
  auto config = py::dict();
  auto RPN_BBOX_STD_DEV = create_array(layer->blobs[0]);
  auto RPN_ANCHOR_SCALES = create_array(layer->blobs[1]);
  auto RPN_ANCHOR_RATIOS = create_array(layer->blobs[2]);
  auto BACKBONE_STRIDES = create_array(layer->blobs[3]);
  auto IMAGE_SHAPE = create_array(layer->blobs[4]);
  auto RPN_ANCHOR_STRIDE = create_array(layer->blobs[5]);
  config["RPN_BBOX_STD_DEV"] = RPN_BBOX_STD_DEV;
  config["RPN_ANCHOR_SCALES"] = RPN_ANCHOR_SCALES;
  config["RPN_ANCHOR_RATIOS"] = RPN_ANCHOR_RATIOS;
  config["BACKBONE_STRIDES"] = BACKBONE_STRIDES;
  config["IMAGE_SHAPE"] = IMAGE_SHAPE;
  config["RPN_ANCHOR_STRIDE"] = RPN_ANCHOR_STRIDE;

  py::module sys = py::module::import("sys");
  py::list syspath = sys.attr("path");
  syspath.append("nn/third_party/rcnn");
  py::module rpn = py::module::import("rpn");
  py::array anchors_ = rpn.attr("generate_pyramid_anchors")(config);
  py::buffer_info arr_info = anchors_.request();
  float* data = (float*)arr_info.ptr;
  *anchors = (float*)malloc(arr_info.size*sizeof(float));
  if(NULL != *anchors) {
    std::copy(data, data+arr_info.size, *anchors);
    NNDDO(NN_DEBUG, rte_save_raw("tmp/anchors.raw", *anchors, arr_info.size*sizeof(float)));
  } else {
    r = NN_E_NO_MEMORY;
  }

  return r;
}
