/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */

__kernel void input(
		__global const float *in,
		__write_only image2d_t out,
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

	for(n=0; n<N; n++) {
		for(c=0; c<C; c+=4)
		{
			offset = n*H*W*C + y*W*C+x*C+c;
			
			if((C-c) > 3) 
			{
				value = (float4)(in[offset], in[offset+1], in[offset+2], in[offset+3]);
			}
			else if((C-c) > 2)
			{
				value = (float4)(in[offset], in[offset+1], in[offset+2], 0);
			}
			else if((C-c) > 1)
			{
				value = (float4)(in[offset], in[offset+1], 0, 0);
			}
			else
			{
				value = (float4)(in[offset], 0, 0, 0);
			}
#if 0
			printf("input (%d,%d,%d,%d)=[%.2f %.2f %.2f %.2f] -> (%d,%d)\n",
					n, x, y, c,
					value.x, value.y, value.z, value.w,
					x*channels+(c/4), y+n*H);
#endif
			write_imagef(out, (int2)(x*channels+(c/4), y+n*H), value);
		}
	}
}
