/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void dense(
		__read_only image2d_t in,
		__read_only image2d_t weights,
		__read_only image2d_t bias,
		__write_only image2d_t out,
		const int ch_im_in,
		const int N,
		const int ch_im_out)
{
	int n;
	int c = get_global_id(2);

	int in_channels = (ch_im_in+3)>>2;

	int in_c;

	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	float4 out0;
	float4 value;
	float4 weight;

	for(n=0; n<N; n++) {
		out0 = read_imagef(bias, sampler, (int2)(0, c));
		for(in_c=0; in_c<ch_im_in; in_c+=4)
		{
			value = read_imagef(in, sampler, (int2)((in_c/4), n));
			weight = read_imagef(weights, sampler, (int2)((c*4)*in_channels+(in_c/4), n));
			out0.x += weight.x*value.x;
			out0.x += weight.y*value.y;
			out0.x += weight.z*value.z;
			out0.x += weight.w*value.w;

			if((ch_im_out-c*4) > 1)
			{
				weight = read_imagef(weights, sampler, (int2)((c*4+1)*in_channels+(in_c/4), n));
				out0.y += weight.x*value.x;
				out0.y += weight.y*value.y;
				out0.y += weight.z*value.z;
				out0.y += weight.w*value.w;
			}

			if((ch_im_out-c*4) > 2)
			{
				weight = read_imagef(weights, sampler, (int2)((c*4+2)*in_channels+(in_c/4), n));
				out0.z += weight.x*value.x;
				out0.z += weight.y*value.y;
				out0.z += weight.z*value.z;
				out0.z += weight.w*value.w;
			}

			if((ch_im_out-c*4) > 3)
			{
				weight = read_imagef(weights, sampler, (int2)((c*4+3)*in_channels+(in_c/4), n));
				out0.w += weight.x*value.x;
				out0.w += weight.y*value.y;
				out0.w += weight.z*value.z;
				out0.w += weight.w*value.w;
			}
		}
		write_imagef(out, (int2)(c, n), out0);
	}
}