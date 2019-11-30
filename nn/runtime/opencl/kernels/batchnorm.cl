/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void batchnorm(
		__read_only image2d_t in,
		__read_only image2d_t scale,
		__read_only image2d_t bias,
		__read_only image2d_t mean,
		__read_only image2d_t var,
		__write_only image2d_t out,
		const float epsilon,
		const int nC4)
{
	int x = get_global_id(0);
	int y = get_global_id(1);
	int c = x%nC4;

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	float4 in0 = read_imagef(in, sampler, (int2)(x,y));
	float4 scale0 = read_imagef(scale, sampler, (int2)(c, 0));
	float4 bias0 = read_imagef(bias, sampler, (int2)(c, 0));
	float4 mean0 = read_imagef(mean, sampler, (int2)(c, 0));
	float4 var0 = read_imagef(var, sampler, (int2)(c, 0));
	
	float4 out0 = scale0*(in0-mean0)*rsqrt(var0 + (float4)epsilon) + bias0;

	write_imagef(out, (int2)(x,y), out0);
}