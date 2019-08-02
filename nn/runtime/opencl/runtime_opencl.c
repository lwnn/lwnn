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
static const layer_ops_t cl_lops[] =
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

	if(layer->op < ARRAY_SIZE(cl_lops))
	{
		r = cl_lops[layer->op].execute(nn, layer);
	}

	return r;
}

static int cl_init_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;

	if(layer->op < ARRAY_SIZE(cl_lops))
	{
		r = cl_lops[layer->op].init(nn, layer);
	}

	return r;
}

static int cl_deinit_layer(const nn_t* nn, const layer_t* layer)
{
	if(layer->op < ARRAY_SIZE(cl_lops))
	{
		cl_lops[layer->op].deinit(nn, layer);
	}

	return 0;
}

static int cl_create_kernel(const nn_t* nn,
		const char* program, const char* kernel,
		cl_program *clprogram,
		cl_kernel *clkernel)
{
	int r = 0;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
	cl_int errNum;

	*clprogram = cl_create_program(rt->context, rt->device, program);
	if(NULL != (*clprogram))
	{
		*clkernel = clCreateKernel(*clprogram, kernel, &errNum);

		if((NULL == (*clkernel)) || (CL_SUCCESS != errNum))
		{
			NNLOG(NN_ERROR,("CL create kernel %s failed with %d\n", kernel, errNum));
			clReleaseProgram(*clprogram);
			*clprogram = NULL;
			r = NN_E_CREATE_CL_KERNEL_FAILED;
		}
	}
	else
	{
		*clkernel = NULL;
		r = NN_E_CREATE_CL_PROGRAM_FAILED;
	}

	return r;
}

static int cl_set_kernel_args_v(cl_kernel kernel, uint32_t nhwcMask, NHWC_t* nhwc, size_t num, va_list valist)
{
	int r = 0;
	int errNum = CL_SUCCESS;
	int i;
	size_t sz;
	void* arg;

	for(i = 0; (i < num) && (CL_SUCCESS == errNum); i++)
	{
		sz = va_arg(valist, size_t);
		arg =  va_arg(valist, void*);
		errNum = clSetKernelArg(kernel, i, sz, arg);
	}

	if(nhwcMask != 0)
	{
		int *dims = (int*)nhwc;
		int d;

		for(d=0; (d < 4) && (nhwcMask&(1<<d)) && (CL_SUCCESS == errNum); d++)
		{
			errNum = clSetKernelArg(kernel, i, sizeof(int), &dims[d]);
			i++;
		}
	}

	if(CL_SUCCESS != errNum)
	{
		NNLOG(NN_ERROR,("CL set args[%d] failed with %d\n", i, errNum));
		r = NN_E_CL_SET_ARGS_FAILED;
	}

	return r;
}

static int cl_set_kernel_args(cl_kernel kernel, uint32_t nhwcMask, NHWC_t* nhwc, size_t num, ...)
{
	int r = 0;
	va_list valist;

	va_start(valist, num);
	r = cl_set_kernel_args_v(kernel, nhwcMask, nhwc, num, valist);
	va_end(valist);

	return r;
}

static int cl_enqueue_kernel(const nn_t* nn, cl_kernel kernel, NHWC_t* nhwc, int use_cl_hw, int run)
{
	int r = 0;
	cl_int errNum;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	size_t globalWorkSize[2];
	static const size_t localWorkSize[2] = { 1, 1 };

	if(use_cl_hw)
	{
		globalWorkSize[0] = RTE_CL_NHWC_W(*nhwc);
		globalWorkSize[1] = RTE_CL_NHWC_H(*nhwc);
	}
	else
	{
		globalWorkSize[0] = nhwc->W;
		globalWorkSize[1] = nhwc->H;
	};

	errNum = clEnqueueNDRangeKernel(rt->command_queue, kernel, 2, NULL,
									globalWorkSize, localWorkSize,
									0, NULL, NULL);
	if(CL_SUCCESS != errNum)
	{
		r = NN_E_CL_EXECUTE_FAILED;
		NNLOG(NN_ERROR,("CL enqueue failed with %d\n", errNum));
	}
	else
	{
		if(TRUE == run)
		{
			clFlush(rt->command_queue);
			clFinish(rt->command_queue);
		}
	}

	return r;
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

cl_mem rte_cl_create_buffer(const nn_t* nn, size_t sz, const float* init_value)
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
					sizeof(float) * sz, (void*)init_value, &errNum);
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
	cl_image_desc desc;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	fmt.image_channel_order = CL_RGBA;
	fmt.image_channel_data_type = CL_FLOAT;

	memset(&desc, 0, sizeof(desc));
	desc.image_type = CL_MEM_OBJECT_IMAGE2D;
	desc.image_width = W;
	desc.image_height = H;

#if 0
	img2d = clCreateImage2D(rt->context, CL_MEM_READ_WRITE,
						&fmt, W, H, 0, NULL, &errNum);
#else
	img2d = clCreateImage(rt->context, CL_MEM_READ_WRITE,
						&fmt, &desc, NULL, &errNum);
#endif
	if(errNum != CL_SUCCESS)
	{
		NNLOG(NN_ERROR,("CL create image2d(%dx%d) failed with %d\n", H, W, errNum));
		img2d = NULL;
	}

	return img2d;
}

int rte_cl_image2d_copy_in(const nn_t* nn, cl_mem img2d, const float* in, NHWC_t* nhwc)
{
	int r = 0;
	cl_program program;
	cl_kernel kernel;
	cl_mem inm;

	r = cl_create_kernel(nn, OPENCL_PATH "input.cl", "input", &program, &kernel);
	if(0 == r)
	{
		inm = rte_cl_create_buffer(nn, NHWC_SIZE(*nhwc), in);

		if(NULL != inm)
		{
			r = cl_set_kernel_args(kernel, RTE_CL_ARGS_WITH_NHWC, nhwc, 2,
							sizeof(cl_mem), &inm,
							sizeof(cl_mem), &img2d);

			if(0 == r)
			{
				r = cl_enqueue_kernel(nn, kernel, nhwc, FALSE, TRUE);
			}

			clReleaseMemObject(inm);
		}

		clReleaseProgram(program);
		clReleaseKernel(kernel);
	}

	return r;
}

int rte_cl_image2d_copy_out(const nn_t* nn, cl_mem img2d, float* out, NHWC_t* nhwc)
{
	int r = 0;
	cl_program program;
	cl_kernel kernel;
	cl_mem outm;

	r = cl_create_kernel(nn, OPENCL_PATH "output.cl", "output", &program, &kernel);
	if(0 == r)
	{
		outm = rte_cl_create_buffer(nn, NHWC_SIZE(*nhwc), NULL);

		if(NULL != outm)
		{
			r = cl_set_kernel_args(kernel, RTE_CL_ARGS_WITH_NHWC, nhwc, 2,
							sizeof(cl_mem), &img2d,
							sizeof(cl_mem), &outm);

			if(0 == r)
			{
				r = cl_enqueue_kernel(nn, kernel, nhwc, FALSE, TRUE);
			}

			if(0 == r)
			{
				r = rte_cl_read_buffer(nn, outm, out, NHWC_SIZE(*nhwc));
			}

			clReleaseMemObject(outm);
		}

		clReleaseProgram(program);
		clReleaseKernel(kernel);
	}

	return r;
}

cl_mem rte_cl_create_image2d_from_blob(const nn_t* nn, const layer_blob_t* blob)
{
	cl_mem img2d = NULL;
	int r = 0;
	const int* dims = blob->dims;
	NHWC_t nhwc;


	r = NHWC_from(blob->dims, &nhwc);

	NNLOG(NN_DEBUG,("cl create blob: [%dx%dx%dx%d] -> [1x%dx%dx4]\n",
			nhwc.N, nhwc.H, nhwc.W, nhwc.C,
			RTE_CL_NHWC_H(nhwc), RTE_CL_NHWC_W(nhwc)));

	if(0 == r)
	{
		img2d = rte_cl_create_image2d(nn,
					RTE_CL_NHWC_H(nhwc),
					RTE_CL_NHWC_W(nhwc));
		if(NULL != img2d)
		{
			r = rte_cl_image2d_copy_in(nn, img2d, (const float*)blob->blob, &nhwc);

			if(0 != r)
			{
				rte_cl_destory_memory(img2d);
				img2d = NULL;
			}
		}
	}

	return img2d;
}

void rte_cl_destory_memory(cl_mem mem)
{
	clReleaseMemObject(mem);
}

int rte_cl_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			const char* program, const char* kernel,
			size_t sz, size_t nout)
{
	int r = 0;
	cl_int errNum;
	layer_cl_context_t* context = NULL;

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
		r = cl_create_kernel(nn, program, kernel, &context->program, &context->kernel);
	}

	if(0 != r)
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
	va_list valist;
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;

	va_start(valist, num);
	r = cl_set_kernel_args_v(context->kernel, nhwc, &context->nhwc, num, valist);
	va_end(valist);

	if(0 != r)
	{
		NNLOG(NN_ERROR,("CL set args for %s failed\n", layer->name));
	}

	return r;
}

int rte_cl_execute_layer(const nn_t* nn, const layer_t* layer, int use_cl_hw)
{
	int r = 0;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;

	r = cl_enqueue_kernel(nn, context->kernel, &context->nhwc, use_cl_hw, 0);

	if(0 != r)
	{
		NNLOG(NN_ERROR,("CL execute for %s failed\n", layer->name));
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

int rte_cl_create_layer_common(const nn_t* nn, const layer_t* layer,
		const char* program, const char* kernel, size_t ctx_sz)
{
	int r = 0;

	layer_cl_context_t* context;

	r = rte_cl_create_layer_context(nn, layer,
				program, kernel, ctx_sz, 1);

	if(0 == r)
	{
		context = (layer_cl_context_t*)layer->C->context;

		RTE_CL_LOG_LAYER_SHAPE(layer);

		context->out[0] = rte_cl_create_image2d(nn,
					RTE_CL_NHWC_H(context->nhwc),
					RTE_CL_NHWC_W(context->nhwc));

		if(NULL == context->out[0])
		{
			r = NN_E_NO_MEMORY;
			rte_cl_destory_layer_context(nn, layer);
		}
	}

	return r;
}
#endif /* DISABLE_RUNTIME_OPENCL */
