/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#define ALG_MAX(a, b) fmax((a), (b))
#define ALG_MIN(a, b) fmin((a), (b))
#define ALG_ADD(a, b) (a) + (b)
#define ALG_SUB(a, b) (a) - (b)
#define ALG_MUL(a, b) (a) * (b)

#define DEF_ELTWISE(func, op)	\
__kernel void func(		\
		__read_only image2d_t A,	\
		__read_only image2d_t B,	\
		__write_only image2d_t out)	\
{	\
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;	\
	\
	int x = get_global_id(0);	\
	int y = get_global_id(1);	\
	\
	float4 a = read_imagef(A, sampler, (int2)(x,y));	\
	float4 b = read_imagef(B, sampler, (int2)(x,y));	\
	float4 value = ALG_##op(a,b);	\
	\
	write_imagef(out, (int2)(x,y), value);	\
}

DEF_ELTWISE(maximum, MAX)
DEF_ELTWISE(add, ADD)
DEF_ELTWISE(minimum, MIN)
DEF_ELTWISE(mul, MUL)

#define DEF_BROADCAST_ONE(func, op)	\
__kernel void func##_broadcast_one(	\
		__read_only image2d_t A,	\
		__read_only image2d_t B,	\
		__write_only image2d_t out)	\
{	\
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;	\
	\
	int x = get_global_id(0);	\
	int y = get_global_id(1);	\
	\
	float4 a = read_imagef(A, sampler, (int2)(x,y));	\
	float4 b = read_imagef(B, sampler, (int2)(0,0));	\
	float one = b.x;	\
	float4 value = ALG_##op(a,one);	\
	\
	write_imagef(out, (int2)(x,y), value);	\
}

DEF_BROADCAST_ONE(maximum, MAX)
DEF_BROADCAST_ONE(add, ADD)
DEF_BROADCAST_ONE(minimum, MIN)
DEF_BROADCAST_ONE(mul, MUL)

#define DEF_BROADCAST_CHANNEL(func, op)	\
__kernel void func##_broadcast_channel(	\
		__read_only image2d_t A,	\
		__read_only image2d_t B,	\
		__write_only image2d_t out,	\
		const int nC4)	\
{	\
	const sampler_t sampler = CLK_NORMALIZED_COORDS_FALSE | CLK_ADDRESS_CLAMP | CLK_FILTER_NEAREST;	\
	\
	int x = get_global_id(0);	\
	int y = get_global_id(1);	\
	int c = x%nC4;	\
	\
	float4 a = read_imagef(A, sampler, (int2)(x,y));	\
	float4 b = read_imagef(B, sampler, (int2)(c, 0));	\
	float4 value = ALG_##op(a,b);	\
	\
	write_imagef(out, (int2)(x,y), value);	\
}

DEF_BROADCAST_CHANNEL(maximum, MAX)
DEF_BROADCAST_CHANNEL(add, ADD)
DEF_BROADCAST_CHANNEL(minimum, MIN)
DEF_BROADCAST_CHANNEL(mul, MUL)