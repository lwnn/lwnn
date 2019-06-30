/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void output(
		__read_only image2d_t in,
		__global float *out,
		const int N,
		const int H,
		const int W,
		const int C )
{

	int n;
	int c;
	int channels = (C+3)>>2;
	int offset;
	float4 value;
	int x = get_global_id(0);
	int y = get_global_id(1);
	
	sampler_t  sampler =  CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;

	for(n=0; n<N; n++) {
		for(c=0; c<C; c+=4)
		{
			offset = n*H*W*C + y*W*C+x*C+c;
			
			value = read_imagef(in, sampler, (int2)(x*channels+(c/4), y+n*H));
			
			if((C-c) > 3) 
			{
				out[offset] = value.x;
				out[offset+1] = value.y;
				out[offset+2] = value.z;
				out[offset+3] = value.w;
			}
			else if((C-c) > 2)
			{
				out[offset] = value.x;
				out[offset+1] = value.y;
				out[offset+2] = value.z;
			}
			else if((C-c) > 1)
			{
				out[offset] = value.x;
				out[offset+1] = value.y;
			}
			else
			{
				out[offset] = value.x;
			}
#if 0
			printf("output (%d,%d)=[%.2f %.2f %.2f %.2f] -> (%d,%d,%d,%d)\n",
					x*channels+(c/4), y+n*H,
					value.x, value.y, value.z, value.w,
					n, x, y, c);
#endif
			
		}
	}
}
