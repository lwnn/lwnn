// Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
#include "nn.h"
#include <math.h>
#include <vector>

extern "C" void rte_save_raw(const char* name, void* data, size_t sz);

// implementation taken from Caffe2
template <typename T>
struct PreCalc {
  int x1;
  int y1;
  int x2;
  int y2;
  T w1;
  T w2;
  T w3;
  T w4;
};

template <typename T>
void pre_calc_for_bilinear_interpolate(
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int iy_upper,
    const int ix_upper,
    T roi_start_h,
    T roi_start_w,
    T bin_size_h,
    T bin_size_w,
    int roi_bin_grid_h,
    int roi_bin_grid_w,
    std::vector<PreCalc<T>>& pre_calc) {
  int pre_calc_index = 0;
  for (int ph = 0; ph < pooled_height; ph++) {
    for (int pw = 0; pw < pooled_width; pw++) {
      for (int iy = 0; iy < iy_upper; iy++) {
        const T yy = roi_start_h + ph * bin_size_h +
            static_cast<T>(iy + .5f) * bin_size_h / static_cast<T>(roi_bin_grid_h); // e.g., 0.5, 1.5
        for (int ix = 0; ix < ix_upper; ix++) {
          const T xx = roi_start_w + pw * bin_size_w +
              static_cast<T>(ix + .5f) * bin_size_w / static_cast<T>(roi_bin_grid_w);

          T x = xx;
          T y = yy;
          // deal with: inverse elements are out of feature map boundary
          if (y < -1.0 || y > height || x < -1.0 || x > width) {
            // empty
            PreCalc<T> pc;
            pc.x1 = 0;
            pc.y1 = 0;
            pc.x2 = 0;
            pc.y2 = 0;
            pc.w1 = 0;
            pc.w2 = 0;
            pc.w3 = 0;
            pc.w4 = 0;
            pre_calc[pre_calc_index] = pc;
            pre_calc_index += 1;
            continue;
          }

          if (y <= 0) {
            y = 0;
          }
          if (x <= 0) {
            x = 0;
          }

          int y_low = (int)y;
          int x_low = (int)x;
          int y_high;
          int x_high;

          if (y_low >= height - 1) {
            y_high = y_low = height - 1;
            y = (T)y_low;
          } else {
            y_high = y_low + 1;
          }

          if (x_low >= width - 1) {
            x_high = x_low = width - 1;
            x = (T)x_low;
          } else {
            x_high = x_low + 1;
          }

          T ly = y - y_low;
          T lx = x - x_low;
          T hy = 1. - ly, hx = 1. - lx;
          T w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;

          // save weights and indices
          PreCalc<T> pc;
          pc.x1 = x_low;
          pc.y1 = y_low;
          pc.x2 = x_high;
          pc.y2 = y_high;
          pc.w1 = w1;
          pc.w2 = w2;
          pc.w3 = w3;
          pc.w4 = w4;
          pre_calc[pre_calc_index] = pc;
          pre_calc_index += 1;
        }
      }
    }
  }
}

template <typename T>
void ROIAlignForward_cpu_kernel(
    const int nthreads,
    const T* bottom_data,
    const T& spatial_scale,
    const int channels,
    const int height,
    const int width,
    const int pooled_height,
    const int pooled_width,
    const int sampling_ratio,
    const T* bottom_rois,
    T* top_data) {
  const int roi_cols = 4;

  int n_rois = nthreads / channels / pooled_width / pooled_height;
  // (n, c, ph, pw) is an element in the pooled output
  // can be parallelized using omp
  // #pragma omp parallel for num_threads(32)
  for (int n = 0; n < n_rois; n++) {
    int index_n = n * channels * pooled_width * pooled_height;

    const T* offset_bottom_rois = bottom_rois + n * roi_cols;

    // Do not using rounding; this implementation detail is critical
    T roi_start_h = offset_bottom_rois[0] * spatial_scale * height;
    T roi_start_w = offset_bottom_rois[1] * spatial_scale * width;
    T roi_end_h = offset_bottom_rois[2] * spatial_scale * height;
    T roi_end_w = offset_bottom_rois[3] * spatial_scale * width;

    // Force malformed ROIs to be 1x1
    T roi_width = std::max(roi_end_w - roi_start_w, (T)1.);
    T roi_height = std::max(roi_end_h - roi_start_h, (T)1.);
    T bin_size_h = static_cast<T>(roi_height) / static_cast<T>(pooled_height);
    T bin_size_w = static_cast<T>(roi_width) / static_cast<T>(pooled_width);

    // We use roi_bin_grid to sample the grid and mimic integral
    int roi_bin_grid_h = (sampling_ratio > 0)
        ? sampling_ratio
        : ceil(roi_height / pooled_height); // e.g., = 2
    int roi_bin_grid_w =
        (sampling_ratio > 0) ? sampling_ratio : ceil(roi_width / pooled_width);

    // We do average (integral) pooling inside a bin
    const T count = roi_bin_grid_h * roi_bin_grid_w; // e.g. = 4

    NNLOG(NN_DEBUG, ("  start=(%.1f,%.1f) h,w=(%.1f,%.1f), bins=(%.1f,%.1f), bing=(%d,%d)\n",
            roi_start_h, roi_start_w, roi_height, roi_width,
            bin_size_h, bin_size_w, roi_bin_grid_h, roi_bin_grid_w));

    // we want to precalculate indices and weights shared by all channels,
    // this is the key point of optimization
    std::vector<PreCalc<T>> pre_calc(
        roi_bin_grid_h * roi_bin_grid_w * pooled_width * pooled_height);
    pre_calc_for_bilinear_interpolate(
        height,
        width,
        pooled_height,
        pooled_width,
        roi_bin_grid_h,
        roi_bin_grid_w,
        roi_start_h,
        roi_start_w,
        bin_size_h,
        bin_size_w,
        roi_bin_grid_h,
        roi_bin_grid_w,
        pre_calc);
    NNDDO(NN_DEBUG, rte_save_raw("tmp/pre_calc.raw", pre_calc.data(), pre_calc.size()*32));

      for (int c = 0; c < channels; c++) {
      int pre_calc_index = 0;

      for (int ph = 0; ph < pooled_height; ph++) {
        for (int pw = 0; pw < pooled_width; pw++) {
          T output_val = 0.;
          for (int iy = 0; iy < roi_bin_grid_h; iy++) {
            for (int ix = 0; ix < roi_bin_grid_w; ix++) {
              PreCalc<T> pc = pre_calc[pre_calc_index];
              output_val += \
                  pc.w1 * bottom_data[(pc.y1*width+pc.x1)*channels+c] + /* pc.pos1 = y_low * width + x_low; */
                  pc.w2 * bottom_data[(pc.y1*width+pc.x2)*channels+c] + /* pc.pos2 = y_low * width + x_high; */
                  pc.w3 * bottom_data[(pc.y2*width+pc.x1)*channels+c] + /* pc.pos3 = y_high * width + x_low; */
                  pc.w4 * bottom_data[(pc.y2*width+pc.x2)*channels+c];  /* pc.pos4 = y_high * width + x_high; */
              pre_calc_index += 1;
            }
          }
          output_val /= count;

          top_data[(ph*pooled_width+pw)*channels+c] = output_val;
        } // for pw
      } // for ph
    } // for c
  } // for n
}

int calc_roi_level(float y1, float x1, float y2, float x2, float image_area, int n_features) {
  int roi_level = -1;
  float level;

  if((x2>x1) && (y2>y1)) {
    level = (x2-x1)*(y2-y1);
    level = sqrt(level) / (224.0 / sqrt(image_area));
    level = log(level) / log(2.0);
    level = std::min((double)n_features-1, std::max(0.0, 2.0+std::round(level)));
    roi_level = level;
  }

  return roi_level;
}

extern "C" int ROIAlign_forward_cpu(float* o, const float* in,
                                    const float* boxes, const int* indices,
                                    NHWC_t* onhwc, NHWC_t* inhwc)
{
  int r = 0;
  int feature_size = NHWC_BATCH_SIZE(*inhwc);
  int roi_size = NHWC_BATCH_SIZE(*onhwc);
  int num_rois = onhwc->N;
  int channels = onhwc->C;
  int pooled_height = onhwc->H;
  int pooled_width = onhwc->W;
  int height = inhwc->H;
  int width = inhwc->W;
  int i;
  const float* feature;

  NNLOG(NN_DEBUG, ("ROTAlign: image_shape=[%d %d %d %d], pool=[%d %d], %d features\n",
              inhwc->N, height, width, channels, pooled_height, pooled_width, num_rois));

  for(i=0; i<num_rois; i++) {
    feature = in+indices[i]*feature_size;
    ROIAlignForward_cpu_kernel(roi_size, feature, 1.0f, channels,
            height, width, pooled_height, pooled_width, 0,
            (float*)&boxes[4*i], o+roi_size*i);
  }

  return r;
}

extern "C" int Pyramid_ROIAlign_forward_cpu(const nn_t* nn, const layer_t* layer)
{
  int r = 0;
  layer_context_t* context = layer->C->context;
  layer_context_t* roi_context = layer->inputs[0]->C->context;
  layer_context_t* meta_context = layer->inputs[1]->C->context;
  const layer_t** features = &layer->inputs[2];
  layer_context_t* feature_context;
  int n_features = 0; while(features[n_features] != NULL) n_features++;
  assert(roi_context->nhwc.N == 1); /* batch mode is not supported */
  int num_rois = roi_context->nhwc.H;
  int channels = context->nhwc.C;
  int pooled_height = context->nhwc.H;
  int pooled_width = context->nhwc.W;
  float* meta = (float*)meta_context->out[0];
  float image_height = meta[2];
  float image_width = meta[3];
  float image_area = image_height*image_width;
  float* boxes = (float*)roi_context->out[0];
  float* feature;
  float y1,x1,y2,x2;
  int i;
  int roi_level;
  int roi_size = channels*pooled_width*pooled_height;
  float* output = (float*)context->out[0];

  context->nhwc.N = num_rois;

  NNLOG(NN_DEBUG, ("execute %s [%d %d %d %d]: image_shape=[%d %d], pool=[%d %d], %d features\n",
       layer->name, L_SHAPES(layer), (int)image_height, (int)image_width,
       pooled_height, pooled_width, n_features));

  for(i=0; i<num_rois; i++) {
    y1 = boxes[4*i+0];
    x1 = boxes[4*i+1];
    y2 = boxes[4*i+2];
    x2 = boxes[4*i+3];
    roi_level = calc_roi_level(y1,x1,y2,x2,image_area, n_features);
    NNLOG(NN_DEBUG, (" ROI @[%.2f,%.2f,%.2f,%.2f] from feature %d [%d %d %d %d]\n",
                    y1, x1, y2, x2, roi_level,
                    L_SHAPES(features[roi_level>0?roi_level:0])));
    if(roi_level >= 0) {
      feature_context = features[roi_level]->C->context;
      feature = (float*)feature_context->out[0];
      ROIAlignForward_cpu_kernel(roi_size, feature, 1.0f, channels,
              feature_context->nhwc.H, feature_context->nhwc.W,
              pooled_height, pooled_width, 0, (float*)&boxes[4*i],
              output+roi_size*i);
    }
  }

  return r;
}
