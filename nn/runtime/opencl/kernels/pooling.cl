/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void maxpool(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int dim_im_in_x,
		const int dim_im_in_y,
		const int dim_kernel_x,
		const int dim_kernel_y,
		const int padding_x,
		const int padding_y,
		const int stride_x,
		const int stride_y,
		const int N,
		const int dim_im_out_y,
		const int ch_im_out)
{
	int n;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int c = get_global_id(2);
	
	int channels = (ch_im_out+3)>>2;
	
	int k_x, k_y;
	
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	float4 res;
	float4 value;
	
	for(n=0; n<N; n++) {
		res = (float4)(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
		for (k_y = y * stride_y - padding_y; k_y < y * stride_y - padding_y + dim_kernel_y; k_y++) {
			for (k_x = x * stride_x - padding_x; k_x < x * stride_x - padding_x + dim_kernel_x; k_x++) {
				if ((k_y >= 0) && (k_x >= 0) && (k_y < dim_im_in_y) && (k_x < dim_im_in_x)) {
					value = read_imagef(in, sampler, (int2)(k_x*channels+c, k_y+n*dim_im_in_y));
					res = fmax(res, value);
				}
			}
		}
		write_imagef(out, (int2)(x*channels+c, y+n*dim_im_out_y), res);
	}
}
