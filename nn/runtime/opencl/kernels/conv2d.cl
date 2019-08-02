/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void conv2d(
		__read_only image2d_t in,
		__read_only image2d_t weights,
		__read_only image2d_t bias,
		__write_only image2d_t out,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int ch_im_in,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		const int N,
		const int dim_im_out_y,
		const int dim_im_out_x,
		const int ch_im_out)
{

}