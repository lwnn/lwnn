/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void softmax(
		__read_only image2d_t in,
		__write_only image2d_t out,
		const int N,
		const int H,
		const int C)
{
	int n;
	int c;
	int c4 = C/4;
	int channels = (C+3)>>2;
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	int x = get_global_id(0);
	int y = get_global_id(1);

	float4 data;
	float sum;
	float max_value;

	for(n=0; n<N; n++) {
		sum = 0;
		max_value = -FLT_MAX;
		for(c=0; c<c4; c++) {
			data = read_imagef(in, sampler, (int2)(c+x*channels, n*H+y));
			max_value = max(data.x, max_value);
			max_value = max(data.y, max_value);
			max_value = max(data.z, max_value);
			max_value = max(data.w, max_value);
		}
		if((C-c*4) > 0) {
			data = read_imagef(in, sampler, (int2)(c+x*channels, n*H+y));
			max_value = max(data.x, max_value);
			if((C-c*4) > 1) {
				max_value = max(data.y, max_value);
			}
			if((C-c*4) > 2) {
				max_value = max(data.z, max_value);
			}
		}

		for(c=0; c<c4; c++) {
			data = read_imagef(in, sampler, (int2)(c+x*channels, n*H+y));
			data = native_exp(data-max_value);
			sum += data.x;
			sum += data.y;
			sum += data.z;
			sum += data.w;
		}
		if((C-c*4) > 0) {
			data = read_imagef(in, sampler, (int2)(c+x*channels, n*H+y));
			data = native_exp(data-max_value);
			sum += data.x;
			if((C-c*4) > 1) {
				sum += data.y;
			}
			if((C-c*4) > 2) {
				sum += data.w;
			}
		}

		for(c=0; c<C; c+=4) {
			data = read_imagef(in, sampler, (int2)(c/4+x*channels, n*H+y));
			data = native_exp(data-max_value) /sum;
			write_imagef(out, (int2)(c/4+x*channels, n*H+y), data);
		}
	}
}