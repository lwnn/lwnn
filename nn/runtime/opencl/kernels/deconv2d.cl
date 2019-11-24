/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

#define NN_MIN(a, b) (((a) < (b)) ? (a) : (b))

int alg_deconv2d_calculate_position(
		int pos,
		int stride,
		int padding,
		int dim_kernel,
		int dim_in,
		int* in_start,
		int* kernel_start,
		int* kernel_end)
{
	int is_zero = 0;
	int of, adj;
	is_zero = 0;
	*in_start = pos/stride;
	of = pos%stride;
	*kernel_start = padding - of;
	if(*kernel_start >= 0) {
		adj = NN_MIN(*in_start, *kernel_start/stride);
		*kernel_start -= adj*stride;
		*in_start -= adj;
	} else {
		adj = -*kernel_start + dim_kernel;
		if(adj<=stride) {
			is_zero = 1;
		} else {
			adj = NN_MIN(dim_in-1-*in_start, adj/stride);
			*kernel_start += adj*stride;
			*in_start += adj;
		}
	}
	of = dim_kernel - 1 - *kernel_start;
	adj = NN_MIN(dim_in-1-*in_start, of/stride);
	*kernel_end = *kernel_start + adj*stride;

	return is_zero;
}

__kernel void deconv2d(
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
	int n;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int c = get_global_id(2);

	int in_row, in_col;
	int kernel_start_x,kernel_end_x;
	int kernel_start_y,kernel_end_y;
	int in_row_start, in_col_start;
	int is_zero, is_zero_y, is_zero_x;

	int knlX,knlY,l,z;

	int in_channels = (ch_im_in+3)>>2;
	int out_channels = (ch_im_out+3)>>2;

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	float4 out0;
	float4 value;
	float4 weight;
	
	is_zero_y = alg_deconv2d_calculate_position(y, stride_y, padding_y, dim_kernel_y,
						dim_im_in_y, &in_row_start, &kernel_start_y, &kernel_end_y);

	is_zero_x = alg_deconv2d_calculate_position(x, stride_x, padding_x, dim_kernel_x,
			dim_im_in_x, &in_col_start, &kernel_start_x, &kernel_end_x);

	is_zero = (is_zero_y || is_zero_x);

	for(n=0; n<N; n++) {
		out0 = read_imagef(bias, sampler, (int2)(c, 0));
		if(0 == is_zero) {
			for (knlY = kernel_start_y, in_row = in_row_start; knlY <= kernel_end_y; knlY+=stride_y, in_row++) {
				for (knlX = kernel_start_x, in_col = in_col_start; knlX <= kernel_end_x; knlX+=stride_x, in_col++) {
					if ((in_row >= 0) && (in_col >= 0) &&
						(in_row < dim_im_in_y) && (in_col < dim_im_in_x)) {
						for (l = 0; l < ch_im_in; l+=4) {
							value = read_imagef(in, sampler, (int2)(in_col*in_channels+(l/4), in_row+n*dim_im_in_y));
							weight = read_imagef(weights, sampler, (int2)(knlX*in_channels+(l/4), knlY+(c*4)*dim_kernel_y));
							out0.x += weight.x*value.x;
							out0.x += weight.y*value.y;
							out0.x += weight.z*value.z;
							out0.x += weight.w*value.w;
							
							if((ch_im_out-c*4) > 1)
							{
								weight = read_imagef(weights, sampler, (int2)(knlX*in_channels+(l/4), knlY+(c*4+1)*dim_kernel_y));
								out0.y += weight.x*value.x;
								out0.y += weight.y*value.y;
								out0.y += weight.z*value.z;
								out0.y += weight.w*value.w;
							}
	
							if((ch_im_out-c*4) > 2)
							{
								weight = read_imagef(weights, sampler, (int2)(knlX*in_channels+(l/4), knlY+(c*4+2)*dim_kernel_y));
								out0.z += weight.x*value.x;
								out0.z += weight.y*value.y;
								out0.z += weight.z*value.z;
								out0.z += weight.w*value.w;
							}
	
							if((ch_im_out-c*4) > 3)
							{
								weight = read_imagef(weights, sampler, (int2)(knlX*in_channels+(l/4), knlY+(c*4+3)*dim_kernel_y));
								out0.w += weight.x*value.x;
								out0.w += weight.y*value.y;
								out0.w += weight.z*value.z;
								out0.w += weight.w*value.w;
							}
						}
					}
				}
			}
		}
		
#ifdef RELU
		out0 = fmax(out0, (float)0);
#endif

#ifdef LEAKY
		out0 = select(0.1 * out0, out0, out0 >= (float)0);
#endif

		write_imagef(out, (int2)(x*out_channels+c, y+n*dim_im_out_y), out0);
	}
}