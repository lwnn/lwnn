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
	int n;
	int x = get_global_id(0);
	int y = get_global_id(1);
	int c = get_global_id(2);
	
	int in_row, in_col;
	
	int knlX,knlY,l,z;
	
	int in_channels = (ch_im_in+3)>>2;
	int out_channels = (ch_im_out+3)>>2;
	
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	float4 out0 = read_imagef(bias, sampler, (int2)(0, c));
	float4 value;
	float4 weight;

	for(n=0; n<N; n++) {
		for (knlY = 0; knlY < dim_kernel_y; knlY++) {
			for (knlX = 0; knlX < dim_kernel_x; knlX++) {
				in_row = stride_y * y + knlY - padding_y;
				in_col = stride_x * x + knlX - padding_x;
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

		write_imagef(out, (int2)(x*out_channels+c, y+n*dim_im_out_y), out0);
	}
}