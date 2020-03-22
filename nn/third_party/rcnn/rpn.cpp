/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* There is crash issues if import numpy twice
 * https://github.com/pybind/pybind11/issues/1439
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include <pybind11/embed.h>
#include <pybind11/numpy.h>
#include <pthread.h>
#include <stdlib.h>
/* ============================ [ MACROS    ] ====================================================== */
namespace py = pybind11;
using namespace py::literals;

#define RPN_LIB_PATH "nn/third_party/rcnn"
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
extern "C" void rte_save_raw(const char* name, void* data, size_t sz);
/* ============================ [ DATAS     ] ====================================================== */
static pthread_once_t _once = PTHREAD_ONCE_INIT;
static py::module _rpn;
/* ============================ [ LOCALS    ] ====================================================== */
template<typename T>
static py::array create_1d_array(T* data, size_t ndim) {
    auto arr = py::array_t<T>({ndim});
    py::buffer_info arr_info = arr.request();
    T* arr_data = (T*) arr_info.ptr;
    std::copy(data, data+arr_info.size, arr_data);
    return arr;
}

template<typename T>
static py::array create_3d_array(T* data, size_t dim0, size_t dim1, size_t dim2) {
    auto arr = py::array_t<T>({dim0, dim1, dim2});
    py::buffer_info arr_info = arr.request();
    T* arr_data = (T*) arr_info.ptr;
    std::copy(data, data+arr_info.size, arr_data);
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
            return create_1d_array<int32_t>((int32_t*)blob->blob, (size_t)dims[0]);
            break;
        case L_DT_FLOAT:
            return create_1d_array<float>((float*)blob->blob, (size_t)dims[0]);
            break;
        default:
            break;
        }
    }

    printf("unsupported blob with dim=%d, dtype=%d\n", dim, blob->dtype);
    assert(0);

    return py::none();
}

static py::object create_array(const layer_t* layer) {
    int dim = 0;
    const int* dims = layer->dims;

    assert(dims != NULL);

    while((dims[dim] != 0) && (dim < 4)) {
        dim ++;
    };

    if(3 == dim) {
        switch(layer->C->context->dtype) {
        case L_DT_FLOAT:
            return create_3d_array<float>((float*)layer->C->context->out[0],
                            (size_t)dims[0], (size_t)dims[1], (size_t)dims[2]);
            break;
        default:
            break;
        }
    }

    printf("unsupported layer with dim=%d, dtype=%d\n", dim, layer->C->context->dtype);
    assert(0);

    return py::none();
}

static void destory_pyenv(void) {
    py::finalize_interpreter();
}

static void setup_pyenv(void) {
    py::initialize_interpreter();
    py::module sys = py::module::import("sys");
    py::list syspath = sys.attr("path");
    syspath.append(RPN_LIB_PATH);
    _rpn = py::module::import("rpn");
    atexit(destory_pyenv);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
extern "C" int rpn_generate_anchors(const nn_t* nn, const layer_t* layer, float** anchors, size_t* n_anchors)
{
  int r = 0;
  if(NULL == getenv("PYTHONHOME")) {
    printf("WARNING: env PYTHONHOME is not set, please set it correctly if see Py_Initialize error\n");
  }
  pthread_once(&_once, setup_pyenv);
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

  py::array anchors_ = _rpn.attr("generate_pyramid_anchors")(config);
  py::buffer_info arr_info = anchors_.request();
  float* data = (float*)arr_info.ptr;
  *anchors = (float*)malloc(arr_info.size*sizeof(float));
  if(NULL != *anchors) {
    std::copy(data, data+arr_info.size, *anchors);
    *n_anchors = (size_t)arr_info.shape[0];
    NNDDO(NN_DEBUG, rte_save_raw("tmp/anchors.raw", *anchors, arr_info.size*sizeof(float)));
  } else {
    r = NN_E_NO_MEMORY;
  }

  return r;
}

extern "C" int rpn_proposal_forward(const nn_t* nn, const layer_t* layer, float* anchors, size_t n_anchors)
{
  int r = 0;
  pthread_once(&_once, setup_pyenv);
  layer_context_t* context = layer->C->context;
  auto RPN_BBOX_STD_DEV = create_array(layer->blobs[0]);
  auto scores = create_array(layer->inputs[0]);
  auto deltas = create_array(layer->inputs[1]);
  auto anchors_ = create_3d_array<float>(anchors, 1, n_anchors, 4);
  float nms_threshold = RTE_FETCH_FLOAT(layer->blobs[6]->blob, 0);

  py::array roi = _rpn.attr("proposal_forward")(RPN_BBOX_STD_DEV, scores, deltas, anchors_, context->nhwc.H, nms_threshold);
  py::buffer_info arr_info = roi.request();

  if((3==arr_info.ndim) && (context->nhwc.N==arr_info.shape[0]) &&
     (context->nhwc.H==arr_info.shape[1]) && (context->nhwc.C==arr_info.shape[2])) {
    float* data = (float*)arr_info.ptr;
    float* O = (float*)context->out[0];
    std::copy(data, data+arr_info.size, O);
  } else {
    r = NN_E_INVALID_DIMENSION;
  }

  return r;
}
