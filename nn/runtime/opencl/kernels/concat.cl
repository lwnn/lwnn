/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void concat_batch(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int offset,
		const int in_batch,
		const int N,
		const int H,
		const int W,
		const int C)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);

	int out_y = y + offset*H;

	float4 data = read_imagef(in, sampler, (int2)(x, y));
	write_imagef(out, (int2)(x, out_y), data);
}

__kernel void concat_height(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int offset,
		const int in_height,
		const int N,
		const int H,
		const int W,
		const int C)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);	/* batch*in_height + rY */

	int out_y = (y/in_height)*H + (y%in_height) + offset;

	float4 data = read_imagef(in, sampler, (int2)(x, y));
	write_imagef(out, (int2)(x, out_y), data);
}

__kernel void concat_width(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int offset,
		const int in_width,
		const int N,
		const int H,
		const int W,
		const int C)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);

	int channels = (C+3)>>2;
	int out_x = x + offset*channels;

	float4 data = read_imagef(in, sampler, (int2)(x, y));
	write_imagef(out, (int2)(out_x, y), data);
}


__kernel void concat_depth(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int offset,
		const int in_depth,
		const int N,
		const int H,
		const int W,
		const int C)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);	/* rX*in_channels+c*/
	int y = get_global_id(1);

	int in_channels = (in_depth+3)>>2;
	int out_channels = (C+3)>>2;
	int out_x = (x/in_channels)*out_channels + (x%in_channels)+(offset/4);
	float4 data = read_imagef(in, sampler, (int2)(x, y));
	write_imagef(out, (int2)(out_x, y), data);
}