/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef _SSD_BBOX_UTIL_HPP_
#define _SSD_BBOX_UTIL_HPP_
/* ============================ [ INCLUDES  ] ====================================================== */
#include <stdint.h>
#include <assert.h>
#include <cmath>  // for std::fabs and std::signbit
#include <map>
#include <string>
#include <iostream>
#include <algorithm>
#include <utility>
#include <vector>
namespace ssd {
using std::vector;
using std::map;
using std::pair;
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
enum ResizeParameter_Resize_mode {
  ResizeParameter_Resize_mode_WARP = 1,
  ResizeParameter_Resize_mode_FIT_SMALL_SIZE = 2,
  ResizeParameter_Resize_mode_FIT_LARGE_SIZE_AND_PAD = 3
};

enum PriorBoxParameter_CodeType {
  PriorBoxParameter_CodeType_CORNER = 1,
  PriorBoxParameter_CodeType_CENTER_SIZE = 2,
  PriorBoxParameter_CodeType_CORNER_SIZE = 3
};

enum EmitConstraint_EmitType {
  EmitConstraint_EmitType_CENTER = 0,
  EmitConstraint_EmitType_MIN_OVERLAP = 1
};

typedef EmitConstraint_EmitType EmitType;
typedef PriorBoxParameter_CodeType CodeType;

class NormalizedBBox
{
public:
	NormalizedBBox() { }
	~NormalizedBBox() { }

	void set_xmin(float value) { xmin_ = value; }
	void set_ymin(float value) { ymin_ = value; }
	void set_xmax(float value) { xmax_ = value; }
	void set_ymax(float value) { ymax_ = value; }
	void set_score(float value) { score_ = value; }
	void set_difficult(bool v) { difficult_ = v; }
	void set_size(float value) { size_ = value; has_size_ = true; }

	void clear_size() { size_ = 0; has_size_ = false; }

	float xmin() const { return xmin_; }
	float ymin() const { return ymin_; }
	float xmax() const { return xmax_; }
	float ymax() const { return ymax_; }
	float score() const { return score_; }
	bool difficult() const { return difficult_; }
	float size() const { return size_; }

	bool has_size() const { return has_size_; }

private:
	float xmin_ = 0;
	float ymin_ = 0;
	float xmax_ = 0;
	float ymax_ = 0;
	bool difficult_ = false;
	float score_ = 0;
	float size_ = 0;
	bool has_size_ = false;
};

class ResizeParameter {
public:
	ResizeParameter() {}
	~ResizeParameter() {}

	ResizeParameter_Resize_mode resize_mode(void) const { return resize_mode_; }

	float height() const { return height_; }
	float width() const { return width_; }
	float height_scale() const { return height_scale_; }
	float width_scale() const { return width_scale_; }

private:
	ResizeParameter_Resize_mode resize_mode_ = ResizeParameter_Resize_mode_WARP;
	float height_ = 0;
	float width_ = 0;
	float height_scale_ = 0;
	float width_scale_ = 0;

};

class EmitConstraint
{
public:
	EmitConstraint() {}
	~EmitConstraint() {}
	float emit_overlap() const { return emit_overlap_; }
	EmitType emit_type() const { return emit_type_; }

private:
	float emit_overlap_ = 2;
	EmitType emit_type_ = EmitConstraint_EmitType_CENTER;
};

typedef map<int, vector<NormalizedBBox> > LabelBBox;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
} /* namespace ssd */
#endif /* _SSD_BBOX_UTIL_HPP_ */
