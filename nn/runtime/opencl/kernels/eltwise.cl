/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void maximum(
		__read_only image2d_t A,
		__read_only image2d_t B,
		__write_only image2d_t out)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);

	float4 a = read_imagef(A, sampler, (int2)(x,y));
	float4 b = read_imagef(B, sampler, (int2)(x,y));
	float4 value = fmax(a,b);

	write_imagef(out, (int2)(x,y), value);
}

__kernel void minimum(
		__read_only image2d_t A,
		__read_only image2d_t B,
		__write_only image2d_t out)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);

	float4 a = read_imagef(A, sampler, (int2)(x,y));
	float4 b = read_imagef(B, sampler, (int2)(x,y));
	float4 value = fmin(a,b);

	write_imagef(out, (int2)(x,y), value);
}

__kernel void add(
		__read_only image2d_t A,
		__read_only image2d_t B,
		__write_only image2d_t out)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);

	float4 a = read_imagef(A, sampler, (int2)(x,y));
	float4 b = read_imagef(B, sampler, (int2)(x,y));
	float4 value = a + b;

	write_imagef(out, (int2)(x,y), value);
}