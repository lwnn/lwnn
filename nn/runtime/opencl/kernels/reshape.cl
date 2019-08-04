/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void reshape(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int in_N,
		const int in_H,
		const int in_W,
		const int in_C,
		const int N,
		const int H,
		const int W,
		const int C)
{
	int n;
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);
	int c = get_global_id(2);

	int in_channels = (in_C+3)>>2;
	int out_channels = (C+3)>>2;
	
	int in_n,in_x,in_y,in_c;
	int offset;
	
	float4 data;

	for(n=0; n<N; n++) {
		offset = n*H*W*C+y*W*C+x*C+c*4;
		in_n = offset/(in_H*in_W*in_C);
		offset = offset%(in_H*in_W*in_C);
		in_y = offset/(in_W*in_C);
		offset = offset%(in_W*in_C);
		in_x = offset/(in_C);
		in_c = offset%(in_C);

		data = read_imagef(in, sampler, (int2)(in_x*in_channels+in_c/4, in_y+n*in_H));
		write_imagef(out, (int2)(x*out_channels+c, y+n*H), data);
	}
}