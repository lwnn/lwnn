/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "runtime_cpu.h"
#include "bbox_util.hpp"
#ifndef __ANDROID__
#include <boost/iterator/counting_iterator.hpp>
#endif

namespace ssd {
/* ============================ [ MACROS    ] ====================================================== */
#define CHECK_EQ(a,b) assert(a == b)
#define CHECK_NE(a,b) assert(a != b)
#define CHECK_LT(a,b) assert(a < b)
#define CHECK_GE(a,b) assert(a >= b)
#define CHECK_GT(a,b) assert(a > b)
#define LOG(level) std::cout
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
		NormalizedBBox* proj_bbox);
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
bool SortBBoxAscend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
	return bbox1.score() < bbox2.score();
}

bool SortBBoxDescend(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
	return bbox1.score() > bbox2.score();
}

template<typename T>
bool SortScorePairAscend(const pair<float, T>& pair1,
		const pair<float, T>& pair2) {
	return pair1.first < pair2.first;
}

// Explicit initialization.
template bool SortScorePairAscend(const pair<float, int>& pair1,
		const pair<float, int>& pair2);
template bool SortScorePairAscend(const pair<float, pair<int, int> >& pair1,
		const pair<float, pair<int, int> >& pair2);

template<typename T>
bool SortScorePairDescend(const pair<float, T>& pair1,
		const pair<float, T>& pair2) {
	return pair1.first > pair2.first;
}

// Explicit initialization.
template bool SortScorePairDescend(const pair<float, int>& pair1,
		const pair<float, int>& pair2);
template bool SortScorePairDescend(const pair<float, pair<int, int> >& pair1,
		const pair<float, pair<int, int> >& pair2);

NormalizedBBox UnitBBox() {
	NormalizedBBox unit_bbox;
	unit_bbox.set_xmin(0.);
	unit_bbox.set_ymin(0.);
	unit_bbox.set_xmax(1.);
	unit_bbox.set_ymax(1.);
	return unit_bbox;
}

bool IsCrossBoundaryBBox(const NormalizedBBox& bbox) {
	return bbox.xmin() < 0 || bbox.xmin() > 1 || bbox.ymin() < 0
			|| bbox.ymin() > 1 || bbox.xmax() < 0 || bbox.xmax() > 1
			|| bbox.ymax() < 0 || bbox.ymax() > 1;
}

void IntersectBBox(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		NormalizedBBox* intersect_bbox) {
	if (bbox2.xmin() > bbox1.xmax() || bbox2.xmax() < bbox1.xmin()
			|| bbox2.ymin() > bbox1.ymax() || bbox2.ymax() < bbox1.ymin()) {
		// Return [0, 0, 0, 0] if there is no intersection.
		intersect_bbox->set_xmin(0);
		intersect_bbox->set_ymin(0);
		intersect_bbox->set_xmax(0);
		intersect_bbox->set_ymax(0);
	} else {
		intersect_bbox->set_xmin(std::max(bbox1.xmin(), bbox2.xmin()));
		intersect_bbox->set_ymin(std::max(bbox1.ymin(), bbox2.ymin()));
		intersect_bbox->set_xmax(std::min(bbox1.xmax(), bbox2.xmax()));
		intersect_bbox->set_ymax(std::min(bbox1.ymax(), bbox2.ymax()));
	}
}

float BBoxSize(const NormalizedBBox& bbox, const bool normalized = true) {
	if (bbox.xmax() < bbox.xmin() || bbox.ymax() < bbox.ymin()) {
		// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return 0;
	} else {
		if (bbox.has_size()) {
			return bbox.size();
		} else {
			float width = bbox.xmax() - bbox.xmin();
			float height = bbox.ymax() - bbox.ymin();
			if (normalized) {
				return width * height;
			} else {
				// If bbox is not within range [0, 1].
				return (width + 1) * (height + 1);
			}
		}
	}
}

template<typename Dtype>
Dtype BBoxSize(const Dtype* bbox, const bool normalized = true) {
	if (bbox[2] < bbox[0] || bbox[3] < bbox[1]) {
		// If bbox is invalid (e.g. xmax < xmin or ymax < ymin), return 0.
		return Dtype(0.);
	} else {
		const Dtype width = bbox[2] - bbox[0];
		const Dtype height = bbox[3] - bbox[1];
		if (normalized) {
			return width * height;
		} else {
			// If bbox is not within range [0, 1].
			return (width + 1) * (height + 1);
		}
	}
}

template float BBoxSize(const float* bbox, const bool normalized);
template double BBoxSize(const double* bbox, const bool normalized);

void ClipBBox(const NormalizedBBox& bbox, NormalizedBBox* clip_bbox) {
	clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), 1.f), 0.f));
	clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), 1.f), 0.f));
	clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), 1.f), 0.f));
	clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), 1.f), 0.f));
	clip_bbox->clear_size();
	clip_bbox->set_size(BBoxSize(*clip_bbox));
	clip_bbox->set_difficult(bbox.difficult());
}

void ClipBBox(const NormalizedBBox& bbox, const float height, const float width,
		NormalizedBBox* clip_bbox) {
	clip_bbox->set_xmin(std::max(std::min(bbox.xmin(), width), 0.f));
	clip_bbox->set_ymin(std::max(std::min(bbox.ymin(), height), 0.f));
	clip_bbox->set_xmax(std::max(std::min(bbox.xmax(), width), 0.f));
	clip_bbox->set_ymax(std::max(std::min(bbox.ymax(), height), 0.f));
	clip_bbox->clear_size();
	clip_bbox->set_size(BBoxSize(*clip_bbox));
	clip_bbox->set_difficult(bbox.difficult());
}

void ScaleBBox(const NormalizedBBox& bbox, const int height, const int width,
		NormalizedBBox* scale_bbox) {
	scale_bbox->set_xmin(bbox.xmin() * width);
	scale_bbox->set_ymin(bbox.ymin() * height);
	scale_bbox->set_xmax(bbox.xmax() * width);
	scale_bbox->set_ymax(bbox.ymax() * height);
	scale_bbox->clear_size();
	bool normalized = !(width > 1 || height > 1);
	scale_bbox->set_size(BBoxSize(*scale_bbox, normalized));
	scale_bbox->set_difficult(bbox.difficult());
}

void OutputBBox(const NormalizedBBox& bbox, const pair<int, int>& img_size,
		const bool has_resize, const ResizeParameter& resize_param,
		NormalizedBBox* out_bbox) {
	const int height = img_size.first;
	const int width = img_size.second;
	NormalizedBBox temp_bbox = bbox;
	if (has_resize && resize_param.resize_mode()) {
		float resize_height = resize_param.height();
		CHECK_GT(resize_height, 0);
		float resize_width = resize_param.width();
		CHECK_GT(resize_width, 0);
		float resize_aspect = resize_width / resize_height;
		int height_scale = resize_param.height_scale();
		int width_scale = resize_param.width_scale();
		float aspect = static_cast<float>(width) / height;

		float padding;
		NormalizedBBox source_bbox;
		switch (resize_param.resize_mode()) {
		case ResizeParameter_Resize_mode_WARP:
			ClipBBox(temp_bbox, &temp_bbox);
			ScaleBBox(temp_bbox, height, width, out_bbox);
			break;
		case ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD:
			if (aspect > resize_aspect) {
				padding = (resize_height - resize_width / aspect) / 2;
				source_bbox.set_xmin(0.);
				source_bbox.set_ymin(padding / resize_height);
				source_bbox.set_xmax(1.);
				source_bbox.set_ymax(1. - padding / resize_height);
			} else {
				padding = (resize_width - resize_height * aspect) / 2;
				source_bbox.set_xmin(padding / resize_width);
				source_bbox.set_ymin(0.);
				source_bbox.set_xmax(1. - padding / resize_width);
				source_bbox.set_ymax(1.);
			}
			ProjectBBox(source_bbox, bbox, &temp_bbox);
			ClipBBox(temp_bbox, &temp_bbox);
			ScaleBBox(temp_bbox, height, width, out_bbox);
			break;
		case ResizeParameter_Resize_mode_FIT_SMALL_SIZE:
			if (height_scale == 0 || width_scale == 0) {
				ClipBBox(temp_bbox, &temp_bbox);
				ScaleBBox(temp_bbox, height, width, out_bbox);
			} else {
				ScaleBBox(temp_bbox, height_scale, width_scale, out_bbox);
				ClipBBox(*out_bbox, height, width, out_bbox);
			}
			break;
		default:
			LOG(FATAL) << "Unknown resize mode.";
		}
	} else {
		// Clip the normalized bbox first.
		ClipBBox(temp_bbox, &temp_bbox);
		// Scale the bbox according to the original image size.
		ScaleBBox(temp_bbox, height, width, out_bbox);
	}
}

void LocateBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
		NormalizedBBox* loc_bbox) {
	float src_width = src_bbox.xmax() - src_bbox.xmin();
	float src_height = src_bbox.ymax() - src_bbox.ymin();
	loc_bbox->set_xmin(src_bbox.xmin() + bbox.xmin() * src_width);
	loc_bbox->set_ymin(src_bbox.ymin() + bbox.ymin() * src_height);
	loc_bbox->set_xmax(src_bbox.xmin() + bbox.xmax() * src_width);
	loc_bbox->set_ymax(src_bbox.ymin() + bbox.ymax() * src_height);
	loc_bbox->set_difficult(bbox.difficult());
}

bool ProjectBBox(const NormalizedBBox& src_bbox, const NormalizedBBox& bbox,
		NormalizedBBox* proj_bbox) {
	if (bbox.xmin() >= src_bbox.xmax() || bbox.xmax() <= src_bbox.xmin()
			|| bbox.ymin() >= src_bbox.ymax()
			|| bbox.ymax() <= src_bbox.ymin()) {
		return false;
	}
	float src_width = src_bbox.xmax() - src_bbox.xmin();
	float src_height = src_bbox.ymax() - src_bbox.ymin();
	proj_bbox->set_xmin((bbox.xmin() - src_bbox.xmin()) / src_width);
	proj_bbox->set_ymin((bbox.ymin() - src_bbox.ymin()) / src_height);
	proj_bbox->set_xmax((bbox.xmax() - src_bbox.xmin()) / src_width);
	proj_bbox->set_ymax((bbox.ymax() - src_bbox.ymin()) / src_height);
	proj_bbox->set_difficult(bbox.difficult());
	ClipBBox(*proj_bbox, proj_bbox);
	if (BBoxSize(*proj_bbox) > 0) {
		return true;
	} else {
		return false;
	}
}

void ExtrapolateBBox(const ResizeParameter& param, const int height,
		const int width, const NormalizedBBox& crop_bbox,
		NormalizedBBox* bbox) {
	float height_scale = param.height_scale();
	float width_scale = param.width_scale();
	if (height_scale > 0 && width_scale > 0
			&& param.resize_mode()
					== ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
		float orig_aspect = static_cast<float>(width) / height;
		float resize_height = param.height();
		float resize_width = param.width();
		float resize_aspect = resize_width / resize_height;
		if (orig_aspect < resize_aspect) {
			resize_height = resize_width / orig_aspect;
		} else {
			resize_width = resize_height * orig_aspect;
		}
		float crop_height = resize_height
				* (crop_bbox.ymax() - crop_bbox.ymin());
		float crop_width = resize_width * (crop_bbox.xmax() - crop_bbox.xmin());
		CHECK_GE(crop_width, width_scale);
		CHECK_GE(crop_height, height_scale);
		bbox->set_xmin(bbox->xmin() * crop_width / width_scale);
		bbox->set_xmax(bbox->xmax() * crop_width / width_scale);
		bbox->set_ymin(bbox->ymin() * crop_height / height_scale);
		bbox->set_ymax(bbox->ymax() * crop_height / height_scale);
	}
}

float JaccardOverlap(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2,
		const bool normalized = true) {
	NormalizedBBox intersect_bbox;
	IntersectBBox(bbox1, bbox2, &intersect_bbox);
	float intersect_width, intersect_height;
	if (normalized) {
		intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin();
		intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin();
	} else {
		intersect_width = intersect_bbox.xmax() - intersect_bbox.xmin() + 1;
		intersect_height = intersect_bbox.ymax() - intersect_bbox.ymin() + 1;
	}
	if (intersect_width > 0 && intersect_height > 0) {
		float intersect_size = intersect_width * intersect_height;
		float bbox1_size = BBoxSize(bbox1);
		float bbox2_size = BBoxSize(bbox2);
		return intersect_size / (bbox1_size + bbox2_size - intersect_size);
	} else {
		return 0.;
	}
}

template<typename Dtype>
Dtype JaccardOverlap(const Dtype* bbox1, const Dtype* bbox2) {
	if (bbox2[0] > bbox1[2] || bbox2[2] < bbox1[0] || bbox2[1] > bbox1[3]
			|| bbox2[3] < bbox1[1]) {
		return Dtype(0.);
	} else {
		const Dtype inter_xmin = std::max(bbox1[0], bbox2[0]);
		const Dtype inter_ymin = std::max(bbox1[1], bbox2[1]);
		const Dtype inter_xmax = std::min(bbox1[2], bbox2[2]);
		const Dtype inter_ymax = std::min(bbox1[3], bbox2[3]);

		const Dtype inter_width = inter_xmax - inter_xmin;
		const Dtype inter_height = inter_ymax - inter_ymin;
		const Dtype inter_size = inter_width * inter_height;

		const Dtype bbox1_size = BBoxSize(bbox1);
		const Dtype bbox2_size = BBoxSize(bbox2);

		return inter_size / (bbox1_size + bbox2_size - inter_size);
	}
}

template float JaccardOverlap(const float* bbox1, const float* bbox2);
template double JaccardOverlap(const double* bbox1, const double* bbox2);

float BBoxCoverage(const NormalizedBBox& bbox1, const NormalizedBBox& bbox2) {
	NormalizedBBox intersect_bbox;
	IntersectBBox(bbox1, bbox2, &intersect_bbox);
	float intersect_size = BBoxSize(intersect_bbox);
	if (intersect_size > 0) {
		float bbox1_size = BBoxSize(bbox1);
		return intersect_size / bbox1_size;
	} else {
		return 0.;
	}
}

bool MeetEmitConstraint(const NormalizedBBox& src_bbox,
		const NormalizedBBox& bbox, const EmitConstraint& emit_constraint) {
	EmitType emit_type = emit_constraint.emit_type();
	if (emit_type == EmitConstraint_EmitType_CENTER) {
		float x_center = (bbox.xmin() + bbox.xmax()) / 2;
		float y_center = (bbox.ymin() + bbox.ymax()) / 2;
		if (x_center >= src_bbox.xmin() && x_center <= src_bbox.xmax()
				&& y_center >= src_bbox.ymin() && y_center <= src_bbox.ymax()) {
			return true;
		} else {
			return false;
		}
	} else if (emit_type == EmitConstraint_EmitType_MIN_OVERLAP) {
		float bbox_coverage = BBoxCoverage(bbox, src_bbox);
		return bbox_coverage > emit_constraint.emit_overlap();
	} else {
		LOG(FATAL) << "Unknown emit type.";
		return false;
	}
}

void EncodeBBox(const NormalizedBBox& prior_bbox,
		const vector<float>& prior_variance, const CodeType code_type,
		const bool encode_variance_in_target, const NormalizedBBox& bbox,
		NormalizedBBox* encode_bbox) {
	if (code_type == PriorBoxParameter_CodeType_CORNER) {
		if (encode_variance_in_target) {
			encode_bbox->set_xmin(bbox.xmin() - prior_bbox.xmin());
			encode_bbox->set_ymin(bbox.ymin() - prior_bbox.ymin());
			encode_bbox->set_xmax(bbox.xmax() - prior_bbox.xmax());
			encode_bbox->set_ymax(bbox.ymax() - prior_bbox.ymax());
		} else {
			// Encode variance in bbox.
			CHECK_EQ(prior_variance.size(), 4);
			for (int i = 0; i < prior_variance.size(); ++i) {
				CHECK_GT(prior_variance[i], 0);
			}
			encode_bbox->set_xmin(
					(bbox.xmin() - prior_bbox.xmin()) / prior_variance[0]);
			encode_bbox->set_ymin(
					(bbox.ymin() - prior_bbox.ymin()) / prior_variance[1]);
			encode_bbox->set_xmax(
					(bbox.xmax() - prior_bbox.xmax()) / prior_variance[2]);
			encode_bbox->set_ymax(
					(bbox.ymax() - prior_bbox.ymax()) / prior_variance[3]);
		}
	} else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
		float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
		CHECK_GT(prior_width, 0);
		float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
		CHECK_GT(prior_height, 0);
		float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
		float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

		float bbox_width = bbox.xmax() - bbox.xmin();
		CHECK_GT(bbox_width, 0);
		float bbox_height = bbox.ymax() - bbox.ymin();
		CHECK_GT(bbox_height, 0);
		float bbox_center_x = (bbox.xmin() + bbox.xmax()) / 2.;
		float bbox_center_y = (bbox.ymin() + bbox.ymax()) / 2.;

		if (encode_variance_in_target) {
			encode_bbox->set_xmin(
					(bbox_center_x - prior_center_x) / prior_width);
			encode_bbox->set_ymin(
					(bbox_center_y - prior_center_y) / prior_height);
			encode_bbox->set_xmax(log(bbox_width / prior_width));
			encode_bbox->set_ymax(log(bbox_height / prior_height));
		} else {
			// Encode variance in bbox.
			encode_bbox->set_xmin(
					(bbox_center_x - prior_center_x) / prior_width
							/ prior_variance[0]);
			encode_bbox->set_ymin(
					(bbox_center_y - prior_center_y) / prior_height
							/ prior_variance[1]);
			encode_bbox->set_xmax(
					log(bbox_width / prior_width) / prior_variance[2]);
			encode_bbox->set_ymax(
					log(bbox_height / prior_height) / prior_variance[3]);
		}
	} else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
		float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
		CHECK_GT(prior_width, 0);
		float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
		CHECK_GT(prior_height, 0);
		if (encode_variance_in_target) {
			encode_bbox->set_xmin(
					(bbox.xmin() - prior_bbox.xmin()) / prior_width);
			encode_bbox->set_ymin(
					(bbox.ymin() - prior_bbox.ymin()) / prior_height);
			encode_bbox->set_xmax(
					(bbox.xmax() - prior_bbox.xmax()) / prior_width);
			encode_bbox->set_ymax(
					(bbox.ymax() - prior_bbox.ymax()) / prior_height);
		} else {
			// Encode variance in bbox.
			CHECK_EQ(prior_variance.size(), 4);
			for (int i = 0; i < prior_variance.size(); ++i) {
				CHECK_GT(prior_variance[i], 0);
			}
			encode_bbox->set_xmin(
					(bbox.xmin() - prior_bbox.xmin()) / prior_width
							/ prior_variance[0]);
			encode_bbox->set_ymin(
					(bbox.ymin() - prior_bbox.ymin()) / prior_height
							/ prior_variance[1]);
			encode_bbox->set_xmax(
					(bbox.xmax() - prior_bbox.xmax()) / prior_width
							/ prior_variance[2]);
			encode_bbox->set_ymax(
					(bbox.ymax() - prior_bbox.ymax()) / prior_height
							/ prior_variance[3]);
		}
	} else {
		LOG(FATAL) << "Unknown LocLossType.";
	}
}

void DecodeBBox(const NormalizedBBox& prior_bbox,
		const vector<float>& prior_variance, const CodeType code_type,
		const bool variance_encoded_in_target, const bool clip_bbox,
		const NormalizedBBox& bbox, NormalizedBBox* decode_bbox) {
	if (code_type == PriorBoxParameter_CodeType_CORNER) {
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to add the offset
			// predictions.
			decode_bbox->set_xmin(prior_bbox.xmin() + bbox.xmin());
			decode_bbox->set_ymin(prior_bbox.ymin() + bbox.ymin());
			decode_bbox->set_xmax(prior_bbox.xmax() + bbox.xmax());
			decode_bbox->set_ymax(prior_bbox.ymax() + bbox.ymax());
		} else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decode_bbox->set_xmin(
					prior_bbox.xmin() + prior_variance[0] * bbox.xmin());
			decode_bbox->set_ymin(
					prior_bbox.ymin() + prior_variance[1] * bbox.ymin());
			decode_bbox->set_xmax(
					prior_bbox.xmax() + prior_variance[2] * bbox.xmax());
			decode_bbox->set_ymax(
					prior_bbox.ymax() + prior_variance[3] * bbox.ymax());
		}
	} else if (code_type == PriorBoxParameter_CodeType_CENTER_SIZE) {
		float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
		CHECK_GT(prior_width, 0);
		float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
		CHECK_GT(prior_height, 0);
		float prior_center_x = (prior_bbox.xmin() + prior_bbox.xmax()) / 2.;
		float prior_center_y = (prior_bbox.ymin() + prior_bbox.ymax()) / 2.;

		float decode_bbox_center_x, decode_bbox_center_y;
		float decode_bbox_width, decode_bbox_height;
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to retore the offset
			// predictions.
			decode_bbox_center_x = bbox.xmin() * prior_width + prior_center_x;
			decode_bbox_center_y = bbox.ymin() * prior_height + prior_center_y;
			decode_bbox_width = exp(bbox.xmax()) * prior_width;
			decode_bbox_height = exp(bbox.ymax()) * prior_height;
		} else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decode_bbox_center_x = prior_variance[0] * bbox.xmin() * prior_width
					+ prior_center_x;
			decode_bbox_center_y = prior_variance[1] * bbox.ymin()
					* prior_height + prior_center_y;
			decode_bbox_width = exp(prior_variance[2] * bbox.xmax())
					* prior_width;
			decode_bbox_height = exp(prior_variance[3] * bbox.ymax())
					* prior_height;
		}

		decode_bbox->set_xmin(decode_bbox_center_x - decode_bbox_width / 2.);
		decode_bbox->set_ymin(decode_bbox_center_y - decode_bbox_height / 2.);
		decode_bbox->set_xmax(decode_bbox_center_x + decode_bbox_width / 2.);
		decode_bbox->set_ymax(decode_bbox_center_y + decode_bbox_height / 2.);
	} else if (code_type == PriorBoxParameter_CodeType_CORNER_SIZE) {
		float prior_width = prior_bbox.xmax() - prior_bbox.xmin();
		CHECK_GT(prior_width, 0);
		float prior_height = prior_bbox.ymax() - prior_bbox.ymin();
		CHECK_GT(prior_height, 0);
		if (variance_encoded_in_target) {
			// variance is encoded in target, we simply need to add the offset
			// predictions.
			decode_bbox->set_xmin(
					prior_bbox.xmin() + bbox.xmin() * prior_width);
			decode_bbox->set_ymin(
					prior_bbox.ymin() + bbox.ymin() * prior_height);
			decode_bbox->set_xmax(
					prior_bbox.xmax() + bbox.xmax() * prior_width);
			decode_bbox->set_ymax(
					prior_bbox.ymax() + bbox.ymax() * prior_height);
		} else {
			// variance is encoded in bbox, we need to scale the offset accordingly.
			decode_bbox->set_xmin(
					prior_bbox.xmin()
							+ prior_variance[0] * bbox.xmin() * prior_width);
			decode_bbox->set_ymin(
					prior_bbox.ymin()
							+ prior_variance[1] * bbox.ymin() * prior_height);
			decode_bbox->set_xmax(
					prior_bbox.xmax()
							+ prior_variance[2] * bbox.xmax() * prior_width);
			decode_bbox->set_ymax(
					prior_bbox.ymax()
							+ prior_variance[3] * bbox.ymax() * prior_height);
		}
	} else {
		LOG(FATAL) << "Unknown LocLossType.";
	}
	float bbox_size = BBoxSize(*decode_bbox);
	decode_bbox->set_size(bbox_size);
	if (clip_bbox) {
		ClipBBox(*decode_bbox, decode_bbox);
	}
}

void DecodeBBoxes(const vector<NormalizedBBox>& prior_bboxes,
		const vector<vector<float> >& prior_variances, const CodeType code_type,
		const bool variance_encoded_in_target, const bool clip_bbox,
		const vector<NormalizedBBox>& bboxes,
		vector<NormalizedBBox>* decode_bboxes) {
	CHECK_EQ(prior_bboxes.size(), prior_variances.size());
	CHECK_EQ(prior_bboxes.size(), bboxes.size());
	int num_bboxes = prior_bboxes.size();
	if (num_bboxes >= 1) {
		CHECK_EQ(prior_variances[0].size(), 4);
	}
	decode_bboxes->clear();
	for (int i = 0; i < num_bboxes; ++i) {
		NormalizedBBox decode_bbox;
		DecodeBBox(prior_bboxes[i], prior_variances[i], code_type,
				variance_encoded_in_target, clip_bbox, bboxes[i], &decode_bbox);
		decode_bboxes->push_back(decode_bbox);
	}
}

void DecodeBBoxesAll(const vector<LabelBBox>& all_loc_preds,
		const vector<NormalizedBBox>& prior_bboxes,
		const vector<vector<float> >& prior_variances, const int num,
		const bool share_location, const int num_loc_classes,
		const int background_label_id, const CodeType code_type,
		const bool variance_encoded_in_target, const bool clip,
		vector<LabelBBox>* all_decode_bboxes) {
	CHECK_EQ(all_loc_preds.size(), num);
	all_decode_bboxes->clear();
	all_decode_bboxes->resize(num);
	for (int i = 0; i < num; ++i) {
		// Decode predictions into bboxes.
		LabelBBox& decode_bboxes = (*all_decode_bboxes)[i];
		for (int c = 0; c < num_loc_classes; ++c) {
			int label = share_location ? -1 : c;
			if (label == background_label_id) {
				// Ignore background class.
				continue;
			}
			if (all_loc_preds[i].find(label) == all_loc_preds[i].end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL) << "Could not find location predictions for label "
						<< label;
			}
			const vector<NormalizedBBox>& label_loc_preds =
					all_loc_preds[i].find(label)->second;
			DecodeBBoxes(prior_bboxes, prior_variances, code_type,
					variance_encoded_in_target, clip, label_loc_preds,
					&(decode_bboxes[label]));
		}
	}
}

template<typename Dtype>
void GetLocPredictions(const Dtype* loc_data, const int num,
		const int num_preds_per_class, const int num_loc_classes,
		const bool share_location, vector<LabelBBox>* loc_preds) {
	loc_preds->clear();
	if (share_location) {
		CHECK_EQ(num_loc_classes, 1);
	}
	loc_preds->resize(num);
	for (int i = 0; i < num; ++i) {
		LabelBBox& label_bbox = (*loc_preds)[i];
		for (int p = 0; p < num_preds_per_class; ++p) {
			int start_idx = p * num_loc_classes * 4;
			for (int c = 0; c < num_loc_classes; ++c) {
				int label = share_location ? -1 : c;
				if (label_bbox.find(label) == label_bbox.end()) {
					label_bbox[label].resize(num_preds_per_class);
				}
				label_bbox[label][p].set_xmin(loc_data[start_idx + c * 4]);
				label_bbox[label][p].set_ymin(loc_data[start_idx + c * 4 + 1]);
				label_bbox[label][p].set_xmax(loc_data[start_idx + c * 4 + 2]);
				label_bbox[label][p].set_ymax(loc_data[start_idx + c * 4 + 3]);
			}
		}
		loc_data += num_preds_per_class * num_loc_classes * 4;
	}
}

// Explicit initialization.
template void GetLocPredictions(const float* loc_data, const int num,
		const int num_preds_per_class, const int num_loc_classes,
		const bool share_location, vector<LabelBBox>* loc_preds);
template void GetLocPredictions(const double* loc_data, const int num,
		const int num_preds_per_class, const int num_loc_classes,
		const bool share_location, vector<LabelBBox>* loc_preds);

template<typename Dtype>
void GetConfidenceScores(const Dtype* conf_data, const int num,
		const int num_preds_per_class, const int num_classes,
		vector<map<int, vector<float> > >* conf_preds) {
	conf_preds->clear();
	conf_preds->resize(num);
	for (int i = 0; i < num; ++i) {
		map<int, vector<float> >& label_scores = (*conf_preds)[i];
		for (int p = 0; p < num_preds_per_class; ++p) {
			int start_idx = p * num_classes;
			for (int c = 0; c < num_classes; ++c) {
				label_scores[c].push_back(conf_data[start_idx + c]);
			}
		}
		conf_data += num_preds_per_class * num_classes;
	}
}

// Explicit initialization.
template void GetConfidenceScores(const float* conf_data, const int num,
		const int num_preds_per_class, const int num_classes,
		vector<map<int, vector<float> > >* conf_preds);
template void GetConfidenceScores(const double* conf_data, const int num,
		const int num_preds_per_class, const int num_classes,
		vector<map<int, vector<float> > >* conf_preds);

template<typename Dtype>
void GetPriorBBoxes(const Dtype* prior_data, const int num_priors,
		vector<NormalizedBBox>* prior_bboxes,
		vector<vector<float> >* prior_variances) {
	prior_bboxes->clear();
	prior_variances->clear();
	for (int i = 0; i < num_priors; ++i) {
		int start_idx = i * 4;
		NormalizedBBox bbox;
		bbox.set_xmin(prior_data[start_idx]);
		bbox.set_ymin(prior_data[start_idx + 1]);
		bbox.set_xmax(prior_data[start_idx + 2]);
		bbox.set_ymax(prior_data[start_idx + 3]);
		float bbox_size = BBoxSize(bbox);
		bbox.set_size(bbox_size);
		prior_bboxes->push_back(bbox);
	}

	for (int i = 0; i < num_priors; ++i) {
		int start_idx = (num_priors + i) * 4;
		vector<float> var;
		for (int j = 0; j < 4; ++j) {
			var.push_back(prior_data[start_idx + j]);
		}
		prior_variances->push_back(var);
	}
}

// Explicit initialization.
template void GetPriorBBoxes(const float* prior_data, const int num_priors,
		vector<NormalizedBBox>* prior_bboxes,
		vector<vector<float> >* prior_variances);
template void GetPriorBBoxes(const double* prior_data, const int num_priors,
		vector<NormalizedBBox>* prior_bboxes,
		vector<vector<float> >* prior_variances);

template<typename Dtype>
void GetDetectionResults(const Dtype* det_data, const int num_det,
		const int background_label_id,
		map<int, map<int, vector<NormalizedBBox> > >* all_detections) {
	all_detections->clear();
	for (int i = 0; i < num_det; ++i) {
		int start_idx = i * 7;
		int item_id = det_data[start_idx];
		if (item_id == -1) {
			continue;
		}
		int label = det_data[start_idx + 1];
		CHECK_NE(background_label_id, label);
		NormalizedBBox bbox;
		bbox.set_score(det_data[start_idx + 2]);
		bbox.set_xmin(det_data[start_idx + 3]);
		bbox.set_ymin(det_data[start_idx + 4]);
		bbox.set_xmax(det_data[start_idx + 5]);
		bbox.set_ymax(det_data[start_idx + 6]);
		float bbox_size = BBoxSize(bbox);
		bbox.set_size(bbox_size);
		(*all_detections)[item_id][label].push_back(bbox);
	}
}

// Explicit initialization.
template void GetDetectionResults(const float* det_data, const int num_det,
		const int background_label_id,
		map<int, map<int, vector<NormalizedBBox> > >* all_detections);
template void GetDetectionResults(const double* det_data, const int num_det,
		const int background_label_id,
		map<int, map<int, vector<NormalizedBBox> > >* all_detections);

void GetTopKScoreIndex(const vector<float>& scores, const vector<int>& indices,
		const int top_k, vector<pair<float, int> >* score_index_vec) {
	CHECK_EQ(scores.size(), indices.size());

	// Generate index score pairs.
	for (int i = 0; i < scores.size(); ++i) {
		score_index_vec->push_back(std::make_pair(scores[i], indices[i]));
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
			SortScorePairDescend<int>);

	// Keep top_k scores if needed.
	if (top_k > -1 && top_k < score_index_vec->size()) {
		score_index_vec->resize(top_k);
	}
}

void GetMaxScoreIndex(const vector<float>& scores, const float threshold,
		const int top_k, vector<pair<float, int> >* score_index_vec) {
	// Generate index score pairs.
	for (int i = 0; i < scores.size(); ++i) {
		if (scores[i] > threshold) {
			score_index_vec->push_back(std::make_pair(scores[i], i));
		}
	}

	// Sort the score pair according to the scores in descending order
	std::stable_sort(score_index_vec->begin(), score_index_vec->end(),
			SortScorePairDescend<int>);

	// Keep top_k scores if needed.
	if (top_k > -1 && top_k < score_index_vec->size()) {
		score_index_vec->resize(top_k);
	}
}

template<typename Dtype>
void GetMaxScoreIndex(const Dtype* scores, const int num, const float threshold,
		const int top_k, vector<pair<Dtype, int> >* score_index_vec) {
	// Generate index score pairs.
	for (int i = 0; i < num; ++i) {
		if (scores[i] > threshold) {
			score_index_vec->push_back(std::make_pair(scores[i], i));
		}
	}

	// Sort the score pair according to the scores in descending order
	std::sort(score_index_vec->begin(), score_index_vec->end(),
			SortScorePairDescend<int>);

	// Keep top_k scores if needed.
	if (top_k > -1 && top_k < score_index_vec->size()) {
		score_index_vec->resize(top_k);
	}
}

template
void GetMaxScoreIndex(const float* scores, const int num, const float threshold,
		const int top_k, vector<pair<float, int> >* score_index_vec);
template
void GetMaxScoreIndex(const double* scores, const int num,
		const float threshold, const int top_k,
		vector<pair<double, int> >* score_index_vec);
#ifndef __ANDROID__
void ApplyNMS(const vector<NormalizedBBox>& bboxes, const vector<float>& scores,
		const float threshold, const int top_k, const bool reuse_overlaps,
		map<int, map<int, float> >* overlaps, vector<int>* indices) {
	// Sanity check.
	CHECK_EQ(bboxes.size(), scores.size());

	// Get top_k scores (with corresponding indices).
	vector<int> idx(boost::counting_iterator<int>(0),
			boost::counting_iterator<int>(scores.size()));
	vector < pair<float, int> > score_index_vec;
	GetTopKScoreIndex(scores, idx, top_k, &score_index_vec);

	// Do nms.
	indices->clear();
	while (score_index_vec.size() != 0) {
		// Get the current highest score box.
		int best_idx = score_index_vec.front().second;
		const NormalizedBBox& best_bbox = bboxes[best_idx];
		if (BBoxSize(best_bbox) < 1e-5) {
			// Erase small box.
			score_index_vec.erase(score_index_vec.begin());
			continue;
		}
		indices->push_back(best_idx);
		// Erase the best box.
		score_index_vec.erase(score_index_vec.begin());

		if (top_k > -1 && indices->size() >= top_k) {
			// Stop if finding enough bboxes for nms.
			break;
		}

		// Compute overlap between best_bbox and other remaining bboxes.
		// Remove a bbox if the overlap with best_bbox is larger than nms_threshold.
		for (vector<pair<float, int> >::iterator it = score_index_vec.begin();
				it != score_index_vec.end();) {
			int cur_idx = it->second;
			const NormalizedBBox& cur_bbox = bboxes[cur_idx];
			if (BBoxSize(cur_bbox) < 1e-5) {
				// Erase small box.
				it = score_index_vec.erase(it);
				continue;
			}
			float cur_overlap = 0.;
			if (reuse_overlaps) {
				if (overlaps->find(best_idx) != overlaps->end()
						&& overlaps->find(best_idx)->second.find(cur_idx)
								!= (*overlaps)[best_idx].end()) {
					// Use the computed overlap.
					cur_overlap = (*overlaps)[best_idx][cur_idx];
				} else if (overlaps->find(cur_idx) != overlaps->end()
						&& overlaps->find(cur_idx)->second.find(best_idx)
								!= (*overlaps)[cur_idx].end()) {
					// Use the computed overlap.
					cur_overlap = (*overlaps)[cur_idx][best_idx];
				} else {
					cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
					// Store the overlap for future use.
					(*overlaps)[best_idx][cur_idx] = cur_overlap;
				}
			} else {
				cur_overlap = JaccardOverlap(best_bbox, cur_bbox);
			}

			// Remove it if necessary
			if (cur_overlap > threshold) {
				it = score_index_vec.erase(it);
			} else {
				++it;
			}
		}
	}
}

void ApplyNMS(const vector<NormalizedBBox>& bboxes, const vector<float>& scores,
		const float threshold, const int top_k, vector<int>* indices) {
	bool reuse_overlap = false;
	map<int, map<int, float> > overlaps;
	ApplyNMS(bboxes, scores, threshold, top_k, reuse_overlap, &overlaps,
			indices);
}

void ApplyNMS(const bool* overlapped, const int num, vector<int>* indices) {
	vector<int> index_vec(boost::counting_iterator<int>(0),
			boost::counting_iterator<int>(num));
	// Do nms.
	indices->clear();
	while (index_vec.size() != 0) {
		// Get the current highest score box.
		int best_idx = index_vec.front();
		indices->push_back(best_idx);
		// Erase the best box.
		index_vec.erase(index_vec.begin());

		for (vector<int>::iterator it = index_vec.begin();
				it != index_vec.end();) {
			int cur_idx = *it;

			// Remove it if necessary
			if (overlapped[best_idx * num + cur_idx]) {
				it = index_vec.erase(it);
			} else {
				++it;
			}
		}
	}
}
#endif /* __ANDROID__ */
void ApplyNMSFast(const vector<NormalizedBBox>& bboxes,
		const vector<float>& scores, const float score_threshold,
		const float nms_threshold, const float eta, const int top_k,
		vector<int>* indices) {
	// Sanity check.
	CHECK_EQ(bboxes.size(), scores.size());

	// Get top_k scores (with corresponding indices).
	vector < pair<float, int> > score_index_vec;
	GetMaxScoreIndex(scores, score_threshold, top_k, &score_index_vec);

	// Do nms.
	float adaptive_threshold = nms_threshold;
	indices->clear();
	while (score_index_vec.size() != 0) {
		const int idx = score_index_vec.front().second;
		bool keep = true;
		for (int k = 0; k < indices->size(); ++k) {
			if (keep) {
				const int kept_idx = (*indices)[k];
				float overlap = JaccardOverlap(bboxes[idx], bboxes[kept_idx]);
				keep = overlap <= adaptive_threshold;
			} else {
				break;
			}
		}
		if (keep) {
			indices->push_back(idx);
		}
		score_index_vec.erase(score_index_vec.begin());
		if (keep && eta < 1 && adaptive_threshold > 0.5) {
			adaptive_threshold *= eta;
		}
	}
}

template<typename Dtype>
void ApplyNMSFast(const Dtype* bboxes, const Dtype* scores, const int num,
		const float score_threshold, const float nms_threshold, const float eta,
		const int top_k, vector<int>* indices) {
	// Get top_k scores (with corresponding indices).
	vector < pair<Dtype, int> > score_index_vec;
	GetMaxScoreIndex(scores, num, score_threshold, top_k, &score_index_vec);

	// Do nms.
	float adaptive_threshold = nms_threshold;
	indices->clear();
	while (score_index_vec.size() != 0) {
		const int idx = score_index_vec.front().second;
		bool keep = true;
		for (int k = 0; k < indices->size(); ++k) {
			if (keep) {
				const int kept_idx = (*indices)[k];
				float overlap = JaccardOverlap(bboxes + idx * 4,
						bboxes + kept_idx * 4);
				keep = overlap <= adaptive_threshold;
			} else {
				break;
			}
		}
		if (keep) {
			indices->push_back(idx);
		}
		score_index_vec.erase(score_index_vec.begin());
		if (keep && eta < 1 && adaptive_threshold > 0.5) {
			adaptive_threshold *= eta;
		}
	}
}

template
void ApplyNMSFast(const float* bboxes, const float* scores, const int num,
		const float score_threshold, const float nms_threshold, const float eta,
		const int top_k, vector<int>* indices);
template
void ApplyNMSFast(const double* bboxes, const double* scores, const int num,
		const float score_threshold, const float nms_threshold, const float eta,
		const int top_k, vector<int>* indices);

extern "C" int detection_output_forward(
		const float* loc_data,
		const float* conf_data,
		const float* prior_data,
		float* top_data,
		int num_priors_,
		float nms_threshold_,
		float confidence_threshold_,
		int num_classes_,
		int share_location_,
		int background_label_id_,
		int top_k_,
		int keep_top_k_,
		CodeType code_type_,
		bool variance_encoded_in_target_,
		int eta_,
		layer_context_t* context
		)
{
	int r = 0;
	int num_loc_classes_ = share_location_ ? 1 : num_classes_;
	int num = context->nhwc.N;

	// Retrieve all location predictions.
	vector<LabelBBox> all_loc_preds;
	GetLocPredictions(loc_data, num, num_priors_, num_loc_classes_,
			share_location_, &all_loc_preds);
	// Retrieve all confidences.
	vector < map<int, vector<float> > > all_conf_scores;
	GetConfidenceScores(conf_data, num, num_priors_, num_classes_,
			&all_conf_scores);

	// Retrieve all prior bboxes. It is same within a batch since we assume all
	// images in a batch are of same dimension.
	vector<NormalizedBBox> prior_bboxes;
	vector < vector<float> > prior_variances;
	GetPriorBBoxes(prior_data, num_priors_, &prior_bboxes, &prior_variances);

	// Decode all loc predictions to bboxes.
	vector<LabelBBox> all_decode_bboxes;
	const bool clip_bbox = false;
	DecodeBBoxesAll(all_loc_preds, prior_bboxes, prior_variances, num,
			share_location_, num_loc_classes_, background_label_id_, code_type_,
			variance_encoded_in_target_, clip_bbox, &all_decode_bboxes);

	int num_kept = 0;
	vector < map<int, vector<int> > > all_indices;
	for (int i = 0; i < num; ++i) {
		const LabelBBox& decode_bboxes = all_decode_bboxes[i];
		const map<int, vector<float> >& conf_scores = all_conf_scores[i];
		map<int, vector<int> > indices;
		int num_det = 0;
		for (int c = 0; c < num_classes_; ++c) {
			if (c == background_label_id_) {
				// Ignore background class.
				continue;
			}
			if (conf_scores.find(c) == conf_scores.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL)<< "Could not find confidence predictions for label " << c;
			}
			const vector<float>& scores = conf_scores.find(c)->second;
			int label = share_location_ ? -1 : c;
			if (decode_bboxes.find(label) == decode_bboxes.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL)<< "Could not find location predictions for label " << label;
				continue;
			}
			const vector<NormalizedBBox>& bboxes =
					decode_bboxes.find(label)->second;
			ApplyNMSFast(bboxes, scores, confidence_threshold_, nms_threshold_,
					eta_, top_k_, &(indices[c]));
			num_det += indices[c].size();
		}
		if (keep_top_k_ > -1 && num_det > keep_top_k_) {
			vector < pair<float, pair<int, int> > > score_index_pairs;
			for (map<int, vector<int> >::iterator it = indices.begin();
					it != indices.end(); ++it) {
				int label = it->first;
				const vector<int>& label_indices = it->second;
				if (conf_scores.find(label) == conf_scores.end()) {
					// Something bad happened for current label.
					LOG(FATAL)<< "Could not find location predictions for " << label;
					continue;
				}
				const vector<float>& scores = conf_scores.find(label)->second;
				for (int j = 0; j < label_indices.size(); ++j) {
					int idx = label_indices[j];
					CHECK_LT(idx, scores.size());
					score_index_pairs.push_back(
							std::make_pair(scores[idx],
									std::make_pair(label, idx)));
				}
			}
			// Keep top k results per image.
			std::sort(score_index_pairs.begin(), score_index_pairs.end(),
					SortScorePairDescend<pair<int, int> >);
			score_index_pairs.resize(keep_top_k_);
			// Store the new indices.
			map<int, vector<int> > new_indices;
			for (int j = 0; j < score_index_pairs.size(); ++j) {
				int label = score_index_pairs[j].second.first;
				int idx = score_index_pairs[j].second.second;
				new_indices[label].push_back(idx);
			}
			all_indices.push_back(new_indices);
			num_kept += keep_top_k_;
		} else {
			all_indices.push_back(indices);
			num_kept += num_det;
		}
	}

	vector<int> top_shape(2, 1);
	top_shape.push_back(num_kept);
	top_shape.push_back(7);

	if(num_kept > (context->nhwc.N*context->nhwc.H)) {
		num_kept =  (context->nhwc.N*context->nhwc.H);
	}

	context->nhwc.N = num_kept;
	context->nhwc.H = 1;

	int count = 0;
	for (int i = 0; i < num; ++i) {
		const map<int, vector<float> >& conf_scores = all_conf_scores[i];
		const LabelBBox& decode_bboxes = all_decode_bboxes[i];
		for (map<int, vector<int> >::iterator it = all_indices[i].begin();
				it != all_indices[i].end(); ++it) {
			int label = it->first;
			if (conf_scores.find(label) == conf_scores.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL)<< "Could not find confidence predictions for " << label;
				continue;
			}
			const vector<float>& scores = conf_scores.find(label)->second;
			int loc_label = share_location_ ? -1 : label;
			if (decode_bboxes.find(loc_label) == decode_bboxes.end()) {
				// Something bad happened if there are no predictions for current label.
				LOG(FATAL)<< "Could not find location predictions for " << loc_label;
				continue;
			}
			const vector<NormalizedBBox>& bboxes =
					decode_bboxes.find(loc_label)->second;
			vector<int>& indices = it->second;
			for (int j = 0; j < indices.size(); ++j) {
				int idx = indices[j];
				top_data[count * 7] = i;
				top_data[count * 7 + 1] = label;
				top_data[count * 7 + 2] = scores[idx];
				const NormalizedBBox& bbox = bboxes[idx];
				top_data[count * 7 + 3] = bbox.xmin();
				top_data[count * 7 + 4] = bbox.ymin();
				top_data[count * 7 + 5] = bbox.xmax();
				top_data[count * 7 + 6] = bbox.ymax();
				++count;

				NNLOG(NN_DEBUG, (" detect B=%d L=%d P=%.2f @ [%.2f %.2f %.2f %.2f]\n",
						i, label, scores[idx],
						bbox.xmin(), bbox.ymin(), bbox.xmax(), bbox.ymax()));
			}
		}
	}

	return r;
}

extern "C" int layer_cpu_float_DETECTIONOUTPUT_execute(const nn_t* nn,
		const layer_t* layer) {
	int r = 0;
	layer_cpu_context_t* context = (layer_cpu_context_t*) layer->C->context;
	layer_cpu_context_t* mbox_loc_context =
			(layer_cpu_context_t*) layer->inputs[0]->C->context;
	layer_cpu_context_t* mbox_conf_context =
			(layer_cpu_context_t*) layer->inputs[1]->C->context;
	const float* loc_data = (float*) mbox_loc_context->out[0];
	const float* conf_data = (float*) mbox_conf_context->out[0];
	const float* prior_data = (float*) layer->blobs[2]->blob;
	float* top_data = (float*)nn_get_output_data(nn, layer);

	int num_priors_ = RTE_FETCH_INT32(layer->blobs[2]->dims,2) / 4;
	float nms_threshold_ = RTE_FETCH_FLOAT(layer->blobs[0]->blob, 0);
	float confidence_threshold_ = RTE_FETCH_FLOAT(layer->blobs[0]->blob, 1);
	int num_classes_ = RTE_FETCH_INT32(layer->blobs[1]->blob, 0);
	int share_location_ = RTE_FETCH_INT32(layer->blobs[1]->blob, 1);
	int background_label_id_ = RTE_FETCH_INT32(layer->blobs[1]->blob, 2);
	int top_k_ = RTE_FETCH_INT32(layer->blobs[1]->blob, 3);
	int keep_top_k_ = RTE_FETCH_INT32(layer->blobs[1]->blob, 4);
	CodeType code_type_ = (CodeType)RTE_FETCH_INT32(layer->blobs[1]->blob, 5);
	int num_loc_classes_ = share_location_ ? 1 : num_classes_;
	bool variance_encoded_in_target_ = false;
	int eta_ = 1.0;

	NNLOG(NN_DEBUG, ("execute %s\n",layer->name));

	if(NULL == top_data)
	{
		r = NN_E_NO_OUTPUT_BUFFER_PROVIDED;
	}
	else
	{
		layer_get_NHWC(layer, &context->nhwc);

		r = detection_output_forward(
			loc_data,
			conf_data,
			prior_data,
			top_data,
			num_priors_,
			nms_threshold_,
			confidence_threshold_,
			num_classes_,
			share_location_,
			background_label_id_,
			top_k_,
			keep_top_k_,
			code_type_,
			variance_encoded_in_target_,
			eta_,
			(layer_context_t*)context);
	}

	return r;
}
} /* namespace ssd */
#endif /* DISABLE_RUNTIME_CPU_FLOAT */
