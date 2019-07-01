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

} rte_cl_t;
/* ============================ [ DECLARES  ] ====================================================== */
#define OP_DEF(op) L_OPS_DECLARE(cl_##op);
#include "opdef.h"
#undef OP_DEF
/* ============================ [ DATAS     ] ====================================================== */
static const layer_ops_t lops[] =
{
#define OP_DEF(op) L_OPS_REF(cl_##op),
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

static void cl_show_build_errors(cl_program program, cl_device_id device)
{
	char *build_log;
	size_t sz = 0;
	cl_int errNum;

	errNum = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &sz);

	if((sz > 0) && (CL_SUCCESS == errNum))
	{
		build_log = malloc(sz+1);
		if(NULL != build_log)
		{
			errNum = clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, sz, build_log, NULL);

			if(CL_SUCCESS == errNum)
			{
				build_log[sz] = '\0';
				NNLOG(NN_ERROR,(build_log));
			}
		}
	}
}

static cl_program cl_create_program(cl_context context, cl_device_id device, const char* fileName)
{
	cl_int errNum = CL_SUCCESS;
	cl_program program = NULL;
	char* srcStr = NULL;
	FILE* file;
	size_t sz;

	NNLOG(NN_DEBUG, ("CL load %s\n", fileName));

	file = fopen(fileName, "rb");

	if(NULL != file)
	{
		fseek(file, 0, SEEK_END);
		sz = ftell(file);
		srcStr = malloc(sz+1);
		fseek(file, 0, SEEK_SET);
		if(NULL != srcStr)
		{
			fread(srcStr, 1, sz, file);
			srcStr[sz] = '\0';
			program = clCreateProgramWithSource(context, 1,
				(const char**)&srcStr,
				&sz, &errNum);
			if(CL_SUCCESS == errNum)
			{
				errNum = clBuildProgram(program, 1, &device, NULL, NULL, NULL);

				if(CL_SUCCESS != errNum)
				{
					cl_show_build_errors(program, device);
					clReleaseProgram(program);
					program = NULL;

					NNLOG(NN_ERROR,("CL build program %s failed with %d\n", fileName, errNum));
				}
			}
			else
			{
				NNLOG(NN_ERROR,("CL create program %s failed with %d\n", fileName, errNum));
			}

			free(srcStr);
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

static int cl_deinit_layer(const nn_t* nn, const layer_t* layer)
{
	if(layer->op < (sizeof(lops)/sizeof(layer_ops_t)))
	{
		lops[layer->op].deinit(nn, layer);
	}

	return 0;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
runtime_t rte_OPENCL_create(const nn_t* nn)
{
	rte_cl_t* rt = NULL;

	rt = malloc(sizeof(rte_cl_t));
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

void rte_OPENCL_destory(const nn_t* nn)
{
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	rte_do_for_each_layer(nn, cl_deinit_layer);

	clReleaseCommandQueue(rt->command_queue);
	clReleaseContext(rt->context);
}

int rte_OPENCL_init(const nn_t* nn)
{
	int r;

	r = rte_do_for_each_layer(nn, cl_init_layer);

	return r;
}

int rte_OPENCL_execute(const nn_t* nn)
{
	int r;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	r =  rte_do_for_each_layer(nn, cl_execute_layer);

	if(0 == r)
	{
		clFlush(rt->command_queue);
		clFinish(rt->command_queue);
	}

	return r;
}

cl_mem rte_cl_create_buffer(const nn_t* nn, size_t sz, float* init_value)
{
	cl_int errNum;
	cl_mem buffer;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
	cl_mem_flags flags = CL_MEM_READ_WRITE;

	if(init_value != NULL)
	{
		flags |= CL_MEM_COPY_HOST_PTR;
	}

	buffer = clCreateBuffer(rt->context, flags,
					sizeof(float) * sz, init_value, &errNum);
	if(errNum != CL_SUCCESS)
	{
		NNLOG(NN_ERROR,("CL create buffer(%d) failed with %d\n", sz, errNum));
		buffer = NULL;
	}

	return buffer;
}

cl_mem rte_cl_create_image2d(const nn_t* nn, int H, int W)
{
	cl_int errNum;
	cl_mem img2d;
	cl_image_format fmt;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	fmt.image_channel_order = CL_RGBA;
	fmt.image_channel_data_type = CL_FLOAT;

	img2d = clCreateImage2D(rt->context, CL_MEM_READ_WRITE,
						&fmt, W, H, 0, NULL, &errNum);

	if(errNum != CL_SUCCESS)
	{
		NNLOG(NN_ERROR,("CL create image2d(%dx%d) failed with %d\n", H, W, errNum));
		img2d = NULL;
	}

	return img2d;
}

int rte_cl_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			const char* program, const char* kernel,
			size_t sz, size_t nout)
{
	int r = 0;
	cl_int errNum;
	layer_cl_context_t* context = NULL;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	assert(sz >= sizeof(layer_cl_context_t));

	context = malloc(sz+nout*sizeof(cl_mem));

	if(context != NULL)
	{
		context->dtype = L_DT_FLOAT;
		context->out = (cl_mem*)(((unsigned long long)context)+sz);
		context->nout = nout;
		r = layer_get_NHWC(layer, &context->nhwc);
		if(0 != r)
		{
			free(context);
		}
	}
	else
	{
		r = NN_E_NO_MEMORY;
	}

	if(0 == r)
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

	if(NULL == context->kernel)
	{
		r = NN_E_CREATE_CL_CONTEXT_FAILED;
		free(context);
	}
	else
	{
		layer->C->context = (layer_context_t*)context;
	}

	return r;
}

void rte_cl_destory_layer_context(const nn_t* nn, const layer_t* layer)
{
	size_t i;
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;

	if(NULL != context)
	{
		clReleaseKernel(context->kernel);
		clReleaseProgram(context->program);

		for(i=0; i<context->nout; i++)
		{
			if(NULL != context->out[i])
			{
				clReleaseMemObject(context->out[i]);
			}
		}

		free(context);
	}

	layer->C->context = NULL;
}

int rte_cl_set_layer_args(
			const nn_t* nn, const layer_t* layer,
			uint32_t nhwc, size_t num, ...)
{
	int r = 0;
	int errNum = CL_SUCCESS;
	int i;
	va_list valist;
	size_t sz;
	void* arg;
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;

	va_start(valist, num);
	for(i = 0; (i < num) && (CL_SUCCESS == errNum); i++)
	{
		sz = va_arg(valist, size_t);
		arg =  va_arg(valist, void*);
		errNum = clSetKernelArg(context->kernel, i, sz, arg);
	}
	va_end(valist);

	if(nhwc != 0)
	{
		int *dims = (int*)&(context->nhwc);
		int d;

		for(d=0; (d < 4) && (nhwc&(1<<d)) && (CL_SUCCESS == errNum); d++)
		{
			errNum = clSetKernelArg(context->kernel, i, sizeof(int), &dims[d]);
			i++;
		}
	}

	if(CL_SUCCESS != errNum)
	{
		r = NN_E_CL_SET_ARGS_FAILED;
		NNLOG(NN_ERROR,("CL set args[%d] for %s failed with %d\n", i, layer->name, errNum));
	}

	return r;
}

int rte_cl_execute_layer(const nn_t* nn, const layer_t* layer, int use_cl_hw)
{
	int r = 0;
	cl_int errNum;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;

	size_t globalWorkSize[2];

	if(use_cl_hw)
	{
		globalWorkSize[0] = RTE_CL_NHWC_W(context->nhwc);
		globalWorkSize[1] = RTE_CL_NHWC_H(context->nhwc);
	}
	else
	{
		globalWorkSize[0] = context->nhwc.W;
		globalWorkSize[1] = context->nhwc.H;
	};
	size_t localWorkSize[2] = { 1, 1 };

	errNum = clEnqueueNDRangeKernel(rt->command_queue, context->kernel, 2, NULL,
									globalWorkSize, localWorkSize,
									0, NULL, NULL);
	if(CL_SUCCESS != errNum)
	{
		r = NN_E_CL_EXECUTE_FAILED;
		NNLOG(NN_ERROR,("CL execute %s failed with %d\n", layer->name, errNum));
	}

	return r;
}

int rte_cl_read_buffer(const nn_t* nn, cl_mem buffer, void* data, size_t sz)
{
	int r = 0;
	cl_int errNum;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	errNum = clEnqueueReadBuffer(rt->command_queue, buffer, CL_TRUE,
						0, sz*sizeof(float), data, 0, NULL, NULL);

	if(CL_SUCCESS != errNum)
	{
		r = NN_E_CL_READ_BUFFER_FAILED;
		NNLOG(NN_ERROR,("CL read buffer failed with %d\n", errNum));
	}

	return r;
}
#endif /* DISABLE_RUNTIME_OPENCL */
