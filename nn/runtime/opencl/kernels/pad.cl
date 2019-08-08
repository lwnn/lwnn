/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void pad(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int padding_top,
		const int padding_bottom,
		const int padding_left,
		const int padding_right,
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

	int channels = (C+3)>>2;
	int in_H = H-padding_top-padding_bottom;
	int in_x,in_y;
	float4 data;

	for(n=0; n<N; n++) {
		
		if( ((y < padding_top) && (y > (H-padding_bottom))) ||
			((x < padding_left) && (x > (W-padding_right))))
		{
			data = (float4)(0,0,0,0);
		}
		else
		{
			in_x = x - padding_left;
			in_y = y - padding_top;
			data = read_imagef(in, sampler, (int2)(in_x*channels+c, in_y+n*in_H));
		}
		write_imagef(out, (int2)(x*channels+c, y+n*H), data);
	}
}