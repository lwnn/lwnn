/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void upsample2d(
		__read_only image2d_t in,
#ifdef WITH_MASK
		__read_only image2d_t in_mask,
#endif
		__write_only image2d_t out,
		const int strideX,
		const int strideY,
		const int C)
{
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;
	const int channles = (C+3)>>2;

	int x = get_global_id(0);
	int y = get_global_id(1);

	int ix = (x/channles/strideX)*channles + (x%channles);
	int iy = y/strideY;

	float4 value = read_imagef(in, sampler, (int2)(ix,iy));
	
#ifdef WITH_MASK
	int dx = (x/channles)%strideX;
	int dy = y%strideY;
	uint offset = dy*strideX + dx;
	uint4 mask = read_imageui(in_mask, sampler, (int2)(ix,iy));
	float4 zero = (float4)(0,0,0,0);
	value = select(zero, value, mask==offset);
#endif

	write_imagef(out, (int2)(x,y), value);
}
