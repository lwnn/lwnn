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
		NNLOG(NN_ERROR, ("Failed to find any OpenCL platforms.\n"));
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
		NNLOG(NN_ERROR, ("No OpenCL devices available."));
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

static cl_program cl_create_program(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum = CL_SUCCESS;
	cl_program program = NULL;
	char* srcStr = NULL;
	FILE* file;
	size_t sz;

	NNLOG(NN_DEBUG, ("CL load %s\n", fileName));

	file = fopen(fileName, "r");

	if(NULL != file)
	{
		fseek(file, 0, SEEK_END);
		sz = ftell(file);
		srcStr = malloc(sz+1);
		fseek(file, 0, SEEK_SET);
		if(NULL != srcStr)
		{
			fread(srcStr, 1, sz, file);
			program = clCreateProgramWithSource(context, 1,
				(const char**)&srcStr,
				NULL, &errNum);
			if(CL_SUCCESS == errNum)
			{
				errNum = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);

				if(CL_SUCCESS != errNum)
				{
					clReleaseProgram(program);
					program = NULL;

					NNLOG(NN_ERROR,("CL build program %s failed with %d\n", fileName, errNum));
				}
			}
			else
			{
				NNLOG(NN_ERROR,("CL create program %s failed with %d\n", fileName, errNum));
			}
		}
		fclose(file);
	}
	else
	{
		NNLOG(NN_ERROR,("CL can't open program %s\n", fileName));
	}

	return program;
}

static int cl_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;

	if(layer->op < (sizeof(lops)/sizeof(layer_ops_t)))
	{
		r = lops[layer->op].execute(nn, layer);
	}

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
		NNLOG(NN_ERROR,("CL create image2d(%dx%d) failed with %d\n", H, W, r));
		img2d = NULL;
	}

	return img2d;
}

void* runtime_opencl_create_context(
			const nn_t* nn, const layer_t* layer,
			const char* program, const char* kernel,
			size_t sz, int *r)
{
	cl_int errNum;
	layer_cl_context_t* context = NULL;
	runtime_opencl_t* rt = (runtime_opencl_t*)nn->runtime;

	assert(sz > sizeof(layer_cl_context_t));

	context = malloc(sz);

	if(context != NULL)
	{
		*r = layer_get_NHWC(layer, &context->nhwc);
		if(0 != *r)
		{
			free(context);
			context = NULL;
		}
	}
	else
	{
		*r = NN_E_NO_MEMORY;
	}

	if(0 == *r)
	{
		context->program = cl_create_program(rt->context, rt->device, program);
		if(NULL != context->program)
		{
			context->kernel = clCreateKernel(context->program, kernel, &errNum);

			if((NULL == context->kernel) || (CL_SUCCESS != errNum))
			{
				NNLOG(NN_ERROR,("CL create kernel %s failed with %d\n", kernel, errNum));

				clReleaseProgram(context->program);
			}
		}
	}


	if(NULL == context->program)
	{
		*r = NN_E_CREATE_CL_CONTEXT_FAILED;
		free(context);
		context = NULL;
	}

	return context;
}
#endif /* DISABLE_RUNTIME_OPENCL */
