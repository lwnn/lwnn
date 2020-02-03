/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include "dnnl.hpp"
/* ============================ [ MACROS    ] ====================================================== */
using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
extern "C" void convolve_HWC_ref_nonsquare(const float * Im_in,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const float * wt,
		const int ch_im_out,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		const float * bias,
		float * Im_out,
		const int dim_im_out_x,
		const int dim_im_out_y,
		layer_activation_type_t act
		)
{
	engine eng(engine::kind::cpu, 0);
	stream s(eng);

	const memory::dim batch = 1;

	/* Note: actually, the LWNN in format NHWC, but should give as NCHW to make it works */
	memory::dims usr_src_tz = {batch, ch_im_in, dim_im_in_y, dim_im_in_x};
	memory::dims usr_weights_tz = {ch_im_out, ch_im_in, dim_kernel_y, dim_kernel_x};
	memory::dims usr_dst_tz = {batch, ch_im_out, dim_im_out_y, dim_im_out_x};

	/* in format NCHW */
	memory::dims conv_src_tz = {batch, ch_im_in, dim_im_in_y, dim_im_in_x};
	memory::dims conv_weights_tz = {ch_im_out, ch_im_in, dim_kernel_y, dim_kernel_x};
	memory::dims conv_bias_tz = {ch_im_out};
	memory::dims conv_dst_tz = {batch, ch_im_out, dim_im_out_y, dim_im_out_x};
	memory::dims conv_strides = {stride_y, stride_x};
	memory::dims conv_padding_l = {padding_y, padding_x};
	int padding_r_y = (dim_im_out_y-1)*stride_y+dim_kernel_y-dim_im_in_y-padding_y;
	int padding_r_x = (dim_im_out_x-1)*stride_x+dim_kernel_x-dim_im_in_x-padding_x;
	memory::dims conv_padding_r = {padding_r_y, padding_r_x};

	auto user_src_memory = memory({{usr_src_tz}, dt::f32, tag::nhwc}, eng, (void*)Im_in);
	auto user_weights_memory
			= memory({{usr_weights_tz}, dt::f32, tag::ohwi}, eng, (void*)wt);
	auto user_bias_memory
			= memory({{conv_bias_tz}, dt::f32, tag::x}, eng, (void*)bias);
	auto user_dst_memory = memory({{usr_dst_tz}, dt::f32, tag::nhwc}, eng, (void*)Im_out);

	auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
	auto conv_bias_md = memory::desc({conv_bias_tz}, dt::f32, tag::any);
	auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
	auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::any);

	auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
			algorithm::convolution_auto, conv_src_md, conv_weights_md,
			conv_bias_md, conv_dst_md, conv_strides, conv_padding_l,
			conv_padding_r);
	auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

	auto conv_src_memory = user_src_memory;
	if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
		conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
		reorder(user_src_memory, conv_src_memory)
				.execute(s, user_src_memory, conv_src_memory);
	}

	auto conv_weights_memory = user_weights_memory;
	if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
		conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
		reorder(user_weights_memory, conv_weights_memory)
				.execute(s, user_weights_memory, conv_weights_memory);
	}

	auto conv_bias_memory = user_bias_memory;
	if (conv_prim_desc.bias_desc() != user_bias_memory.get_desc()) {
		conv_bias_memory = memory(conv_prim_desc.weights_desc(), eng);
		reorder(user_bias_memory, conv_bias_memory)
				.execute(s, user_bias_memory, conv_bias_memory);
	}

	auto conv_dst_memory = user_dst_memory;
	if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
		conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);
	}

	auto conv = convolution_forward(conv_prim_desc);
	conv.execute(s, {{DNNL_ARG_SRC, conv_src_memory},
					{DNNL_ARG_WEIGHTS, conv_weights_memory},
					{DNNL_ARG_BIAS, conv_bias_memory},
					{DNNL_ARG_DST, conv_dst_memory}});

	if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
		reorder(conv_dst_memory, user_dst_memory)
				.execute(s, conv_dst_memory, user_dst_memory);
	}

	s.wait();
}

#if 0
extern "C" void dilated_convolve_HWC_ref_nonsquare(const float * Im_in,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const float * wt,
		const int ch_im_out,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		const int dilation_x,
		const int dilation_y,
		const float * bias,
		float * Im_out,
		const int dim_im_out_x,
		const int dim_im_out_y,
		layer_activation_type_t act
		)
{
	engine eng(engine::kind::cpu, 0);
	stream s(eng);

	const memory::dim batch = 1;

	/* Note: actually, the LWNN in format NHWC, but should give as NCHW to make it works */
	memory::dims usr_src_tz = {batch, ch_im_in, dim_im_in_y, dim_im_in_x};
	memory::dims usr_weights_tz = {ch_im_out, ch_im_in, dim_kernel_y, dim_kernel_x};
	memory::dims usr_dst_tz = {batch, ch_im_out, dim_im_out_y, dim_im_out_x};

	/* in format NCHW */
	memory::dims conv_src_tz = {batch, ch_im_in, dim_im_in_y, dim_im_in_x};
	memory::dims conv_weights_tz = {ch_im_out, ch_im_in, dim_kernel_y, dim_kernel_x};
	memory::dims conv_bias_tz = {ch_im_out};
	memory::dims conv_dst_tz = {batch, ch_im_out, dim_im_out_y, dim_im_out_x};
	memory::dims conv_strides = {stride_y, stride_x};
	memory::dims conv_padding = {padding_y, padding_x};
	memory::dims conv_dilations = {dilation_y, dilation_x};

	auto user_src_memory = memory({{usr_src_tz}, dt::f32, tag::nhwc}, eng, (void*)Im_in);
	auto user_weights_memory
			= memory({{usr_weights_tz}, dt::f32, tag::ohwi}, eng, (void*)wt);
	auto user_bias_memory
			= memory({{conv_bias_tz}, dt::f32, tag::x}, eng, (void*)bias);
	auto user_dst_memory = memory({{usr_dst_tz}, dt::f32, tag::nhwc}, eng, (void*)Im_out);

	auto conv_src_md = memory::desc({conv_src_tz}, dt::f32, tag::any);
	auto conv_bias_md = memory::desc({conv_bias_tz}, dt::f32, tag::any);
	auto conv_weights_md = memory::desc({conv_weights_tz}, dt::f32, tag::any);
	auto conv_dst_md = memory::desc({conv_dst_tz}, dt::f32, tag::any);

	auto conv_desc = convolution_forward::desc(prop_kind::forward_inference,
			algorithm::convolution_auto, conv_src_md, conv_weights_md,
			conv_bias_md, conv_dst_md, conv_strides, conv_dilations, conv_padding,
			conv_padding);
	auto conv_prim_desc = convolution_forward::primitive_desc(conv_desc, eng);

	auto conv_src_memory = user_src_memory;
	if (conv_prim_desc.src_desc() != user_src_memory.get_desc()) {
		conv_src_memory = memory(conv_prim_desc.src_desc(), eng);
		reorder(user_src_memory, conv_src_memory)
				.execute(s, user_src_memory, conv_src_memory);
	}

	auto conv_weights_memory = user_weights_memory;
	if (conv_prim_desc.weights_desc() != user_weights_memory.get_desc()) {
		conv_weights_memory = memory(conv_prim_desc.weights_desc(), eng);
		reorder(user_weights_memory, conv_weights_memory)
				.execute(s, user_weights_memory, conv_weights_memory);
	}

	auto conv_bias_memory = user_bias_memory;
	if (conv_prim_desc.bias_desc() != user_bias_memory.get_desc()) {
		conv_bias_memory = memory(conv_prim_desc.weights_desc(), eng);
		reorder(user_bias_memory, conv_bias_memory)
				.execute(s, user_bias_memory, conv_bias_memory);
	}

	auto conv_dst_memory = user_dst_memory;
	if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
		conv_dst_memory = memory(conv_prim_desc.dst_desc(), eng);
	}

	auto conv = convolution_forward(conv_prim_desc);
	conv.execute(s, {{DNNL_ARG_SRC, conv_src_memory},
					{DNNL_ARG_WEIGHTS, conv_weights_memory},
					{DNNL_ARG_BIAS, conv_bias_memory},
					{DNNL_ARG_DST, conv_dst_memory}});

	if (conv_prim_desc.dst_desc() != user_dst_memory.get_desc()) {
		reorder(conv_dst_memory, user_dst_memory)
				.execute(s, conv_dst_memory, user_dst_memory);
	}

	s.wait();
}
#endif
