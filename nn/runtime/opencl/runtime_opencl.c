/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_OPENCL
#include "runtime_opencl.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	cl_context context;
	cl_device_id device;
	cl_command_queue command_queue;

} runtime_opencl_t;
/* ============================ [ DECLARES  ] ====================================================== */
#define OP_DEF(op) L_OPS_DECLARE(opencl_##op);
#include "opdef.h"
#undef OP_DEF
/* ============================ [ DATAS     ] ====================================================== */
static const layer_ops_t lops[] =
{
#define OP_DEF(op) L_OPS_REF(opencl_##op),
	#include "opdef.h"
#undef OP_DEF
};
/* ============================ [ LOCALS    ] ====================================================== */
static cl_context cl_create_context()
{
	cl_int errNum;
	cl_uint numPlatforms;
	cl_platform_id firstPlatformId;
	cl_context context = NULL;

	errNum = clGetPlatformIDs(1, &firstPlatformId, &numPlatforms);
	if (errNum != CL_SUCCESS || numPlatforms <= 0)
	{
		NN_LOG(NN_ERROR, ("Failed to find any OpenCL platforms.\n"));
	}
	else
	{
		cl_context_properties contextProperties[] =
		{
			CL_CONTEXT_PLATFORM,
			(cl_context_properties)firstPlatformId,
			0
		};
		context = clCreateContextFromType(contextProperties, CL_DEVICE_TYPE_GPU, NULL, NULL, &errNum);
	}

	return context;
}

static cl_command_queue cl_create_command_queue(cl_context context, cl_device_id *device)
{
	cl_int errNum;
	cl_device_id *devices;
	cl_command_queue commandQueue = NULL;
	size_t deviceBufferSize = -1;

	errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, 0, NULL, &deviceBufferSize);

	if (deviceBufferSize <= 0)
	{
		NN_LOG(NN_ERROR, ("No OpenCL devices available."));
	}
	else
	{
		devices = malloc(deviceBufferSize);
		errNum = clGetContextInfo(context, CL_CONTEXT_DEVICES, deviceBufferSize, devices, NULL);

		commandQueue = clCreateCommandQueue(context, devices[0], 0, NULL);

		*device = devices[0];
		free(devices);
	}

	return commandQueue;
}

static int cl_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	NN_LOG(NN_DEBUG, (" CL run %-16s: op=%d\n", layer->name, layer->op));
	return r;
}

static int cl_init_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;

	if(layer->op < (sizeof(lops)/sizeof(layer_ops_t)))
	{
		r = lops[layer->op].init(nn, layer);
	}

	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t runtime_opencl_create(const nn_t* nn)
{
	runtime_opencl_t* rt = NULL;

	rt = malloc(sizeof(runtime_opencl_t));
	if(NULL != rt)
	{
		rt->context = cl_create_context();
	}

	if(NULL == rt->context)
	{
		free(rt);
		rt = NULL;
	}
	else
	{
		rt->command_queue = cl_create_command_queue(rt->context, &rt->device);
	}

	if(NULL == rt->command_queue)
	{
		clReleaseContext(rt->context);
		free(rt);
		rt = NULL;
	}

	return rt;
}

void runtime_opencl_destory(const nn_t* nn)
{
	runtime_opencl_t* rt = (runtime_opencl_t*)nn->runtime;

	clReleaseCommandQueue(rt->command_queue);
	clReleaseContext(rt->context);
}

int runtime_opencl_init(const nn_t* nn)
{
	int r;

	r = runtime_do_for_each_layer(nn, cl_init_layer);

	return r;
}

int runtime_opencl_execute(const nn_t* nn)
{
	return runtime_do_for_each_layer(nn, cl_execute_layer);
}

cl_mem runtime_opencl_create_image2d(const nn_t* nn, int H, int W)
{
	cl_int r;
	cl_mem img2d;
	cl_image_format fmt;
	runtime_opencl_t* rt = (runtime_opencl_t*)nn->runtime;

	fmt.image_channel_order = CL_RGBA;
	fmt.image_channel_data_type = CL_FLOAT;

	img2d = clCreateImage2D(rt->context, CL_MEM_READ_WRITE,
						&fmt, W, H, 0, NULL, &r);

	if(r != CL_SUCCESS)
	{
		NN_LOG(NN_ERROR,("CL create image2d(%dx%d) failed with %d\n", H, W, r));
		img2d = NULL;
	}

	return img2d;
}

#endif /* DISABLE_RUNTIME_OPENCL */
