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
		const int C
		)
{

	int n;
	int c;
	int channels = (c+3)>>2;
	int offset;
	float4 value;
	int x = get_global_id(0);
	int y = get_global_id(1);

	
	for(n=0; n<N; n++) {
		offset = n*H*W*C;

		for(c=0; c<C; c+=4)
		{
			offset += y*W*C+x*C+c;
			value = (float4)(in[offset+0], in[offset+1], in[offset+2], in[offset+3]);

			printf("input (%d,%d)=[%.2f %.2f %.2f %.2f] -> (%d,%d)\n",
					x, y,
					value.x, value.y, value.w, value.z,
					x*channels+(c/4), y+n*H);
			write_imagef(out, (int2)(x*channels+(c/4), y+n*H), value);
		}
	}
}

