/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void maxpool(
		__read_only image2d_t in,
		__write_only image2d_t out,
#ifdef WITH_MASK
		__write_only image2d_t out_mask,
#endif
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

#ifdef WITH_MASK
	uint4 idx;
	uint offset;
#endif
	float4 res;
	float4 value;
	
	for(n=0; n<N; n++) {
		res = (float4)(-FLT_MAX, -FLT_MAX, -FLT_MAX, -FLT_MAX);
#ifdef WITH_MASK
		idx = (uint4)(0, 0, 0, 0);
#endif
		for (k_y = y * stride_y - padding_y; k_y < y * stride_y - padding_y + dim_kernel_y; k_y++) {
			for (k_x = x * stride_x - padding_x; k_x < x * stride_x - padding_x + dim_kernel_x; k_x++) {
				if ((k_y >= 0) && (k_x >= 0) && (k_y < dim_im_in_y) && (k_x < dim_im_in_x)) {
					value = read_imagef(in, sampler, (int2)(k_x*channels+c, k_y+n*dim_im_in_y));
#ifdef WITH_MASK
					offset = (k_y-(y * stride_y - padding_y))*stride_x + \
							(k_x-(x * stride_x - padding_x));
					idx = select(idx, offset, value>res);
#endif
					res = fmax(res, value);
				}
			}
		}
		write_imagef(out, (int2)(x*channels+c, y+n*dim_im_out_y), res);
#ifdef WITH_MASK
		write_imageui(out_mask, (int2)(x*channels+c, y+n*dim_im_out_y), idx);
#endif
	}
}

__kernel void avgpool(
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
		res = (float4)(0, 0, 0, 0);
		for (k_y = y * stride_y - padding_y; k_y < y * stride_y - padding_y + dim_kernel_y; k_y++) {
			for (k_x = x * stride_x - padding_x; k_x < x * stride_x - padding_x + dim_kernel_x; k_x++) {
				if ((k_y >= 0) && (k_x >= 0) && (k_y < dim_im_in_y) && (k_x < dim_im_in_x)) {
					value = read_imagef(in, sampler, (int2)(k_x*channels+c, k_y+n*dim_im_in_y));
					res = res + value;
				}
			}
		}
		res = res/(dim_kernel_y*dim_kernel_x);
		write_imagef(out, (int2)(x*channels+c, y+n*dim_im_out_y), res);
	}
}
