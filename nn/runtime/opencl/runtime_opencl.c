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

	cl_program iprg;
	cl_program oprg;
	cl_kernel iknl;
	cl_kernel oknl;
#ifdef ENABLE_CL_IMAGE_REUSE
	STAILQ_HEAD(rte_cl_image_head,rte_cl_image) images;
#endif
} rte_cl_t;
/* ============================ [ DECLARES  ] ====================================================== */
#define OP_DEF(op) L_OPS_DECLARE(cl_##op);
#include "opdef.h"
#undef OP_DEF
#ifndef DISABLE_NN_DDO
extern void rte_ddo_save(const nn_t* nn, const layer_t* layer);
static int cl_ddo_layer(const nn_t* nn, const layer_t* layer);
#endif
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

static cl_program cl_create_program(cl_context context, cl_device_id device,
		const char* fileName, const char* option)
{
	cl_int errNum = CL_SUCCESS;
	cl_program program = NULL;
	char* srcStr = NULL;
	FILE* file;
	size_t sz;

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
				errNum = clBuildProgram(program, 1, &device, option, NULL, NULL);

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

#ifndef DISABLE_NN_DDO
		NNDDO(NN_DEBUG, cl_ddo_layer(nn, layer));
#endif
	}

	return r;
}
#ifndef DISABLE_NN_DDO
static int cl_ddo_layer(const nn_t* nn, const layer_t* layer)
{
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	if(layer->op != L_OP_OUTPUT)
	{
		clFlush(rt->command_queue);
		clFinish(rt->command_queue);
		NNDDO(NN_DEBUG, rte_ddo_save(nn, layer));
	}
	return 0;
}
#endif

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
		const char* option,
		cl_program *clprogram,
		cl_kernel *clkernel)
{
	int r = 0;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
	cl_int errNum;

	NNLOG(NN_DEBUG, ("CL load %s::%s\n", program, kernel));

	*clprogram = cl_create_program(rt->context, rt->device, program, option);
	if(NULL != (*clprogram))
	{
		*clkernel = clCreateKernel(*clprogram, kernel, &errNum);

		if((NULL == (*clkernel)) || (CL_SUCCESS != errNum))
		{
			NNLOG(NN_ERROR,("CL create kernel %s failed with %d\n", kernel, errNum));
			clReleaseProgram(*clprogram);
			*clprogram = NULL;
			*clkernel = NULL;
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

		for(d=0; (d < 4) && (CL_SUCCESS == errNum); d++)
		{
			if(nhwcMask&(1<<d))
			{
				errNum = clSetKernelArg(kernel, i, sizeof(int), &dims[d]);
				i++;
			}
		}
	}

	if(CL_SUCCESS != errNum)
	{
		NNLOG(NN_ERROR,("CL set args[%d] failed with %d\n", i-1, errNum));
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

static int cl_enqueue_kernel(const nn_t* nn, cl_kernel kernel, NHWC_t* nhwc, rte_cl_global_work_type_t gwt, int run)
{
	int r = 0;
	cl_int errNum;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	size_t globalWorkSize[3] = { 1, 1, 1 };
	static const size_t localWorkSize[3] = { 1, 1, 1 };
	size_t sz = 2;

	switch(gwt)
	{
		case RTE_GWT_W_H:
			globalWorkSize[0] = nhwc->W;
			globalWorkSize[1] = nhwc->H;
			break;
		case RTE_GWT_CL_W_H:
			globalWorkSize[0] = RTE_CL_NHWC_W(*nhwc);
			globalWorkSize[1] = RTE_CL_NHWC_H(*nhwc);
			break;
		case RTE_GWT_W_H_C:
			globalWorkSize[0] = nhwc->W;
			globalWorkSize[1] = nhwc->H;
			globalWorkSize[2] = RTE_CL_NHWC_C(*nhwc);
			sz = 3;
			break;
		default:
			assert(0);
			break;
	}

	errNum = clEnqueueNDRangeKernel(rt->command_queue, kernel, sz, NULL,
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
#ifdef ENABLE_CL_IMAGE_REUSE
#ifndef DISABLE_NN_LOG
static int cl_get_image_id(const nn_t* nn, rte_cl_image_t* image)
{
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
	int imageId = -1;
	int id = -1;
	rte_cl_image_t* i;

	STAILQ_FOREACH(i, &(rt->images), entry)
	{
		id ++;
		if(i == image)
		{
			imageId = id;
			break;
		}
	}

	return imageId;
}
#endif
static int cl_adjust_layer_image(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	int i;
	rte_cl_image_t* image;
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;
	#ifndef DISABLE_RTE_FALLBACK
	if(layer->op != L_OP_YOLO)
	#endif
	{
		for(i=0; i<context->nout; i++)
		{
			image = (rte_cl_image_t*)context->out[i];
			if(NULL != image)
			{
				context->out[i] = image->img;
				NNLOG(NN_DEBUG, (" layer %s out[%d] using image%d\n", layer->name, i, cl_get_image_id(nn, image)));
			}
		}
	}

	return r;
}
#endif /* ENABLE_CL_IMAGE_REUSE */
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
	else
	{
		rt->iprg = NULL;
		rt->oprg = NULL;
		rt->iknl = NULL;
		rt->oknl = NULL;
	}

	return rt;
}

void rte_OPENCL_destory(const nn_t* nn)
{
#ifdef ENABLE_CL_IMAGE_REUSE
	rte_cl_image_t* i;
#endif
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	rte_do_for_each_layer(nn, cl_deinit_layer);

#ifdef ENABLE_CL_IMAGE_REUSE
	while(FALSE == STAILQ_EMPTY(&rt->images))
	{
		i = STAILQ_FIRST(&rt->images);
		STAILQ_REMOVE_HEAD(&rt->images, entry);
		if(i->img != NULL)
		{
			clReleaseMemObject(i->img);
		}
		free(i);
	}
#endif

	if(rt->iknl != NULL)
	{
		clReleaseKernel(rt->iknl);
	}

	if(rt->iprg != NULL)
	{
		clReleaseProgram(rt->iprg);
	}

	if(rt->oknl != NULL)
	{
		clReleaseKernel(rt->oknl);
	}

	if(rt->oprg != NULL)
	{
		clReleaseProgram(rt->oprg);
	}

	clReleaseCommandQueue(rt->command_queue);
	clReleaseContext(rt->context);

	free(rt);
}

int rte_OPENCL_init(const nn_t* nn)
{
	int r;
#ifdef ENABLE_CL_IMAGE_REUSE
	rte_cl_image_t* i;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
#ifndef DISABLE_NN_LOG
	size_t sum = 0;
	size_t imageId = -1;
#endif

	STAILQ_INIT(&(rt->images));
#endif

	r = rte_do_for_each_layer(nn, cl_init_layer);

#ifdef ENABLE_CL_IMAGE_REUSE
	if(0 == r)
	{
		NNLOG(NN_DEBUG, ("Memory Usage:\n"));
		STAILQ_FOREACH(i, &(rt->images), entry)
		{
			i->img = rte_cl_create_image2d(nn, i->H, i->W);
			if(i->img == NULL)
			{
				r = NN_E_NO_MEMORY;
				break;
			}

			#ifndef DISABLE_NN_LOG
			sum += i->H*i->W;
			imageId ++;
			#endif
			NNLOG(NN_DEBUG, (" image%d: %dx%d=%d\n", imageId, i->H, i->W, i->H*i->W));
		}
		NNLOG(NN_DEBUG, (" summary: %d\n", sum));
	}

	if(0 == r)
	{
		r = rte_do_for_each_layer(nn, cl_adjust_layer_image);
	}
#endif

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
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
	cl_mem inm;

	if(NULL == rt->iknl)
	{
		r = cl_create_kernel(nn, OPENCL_PATH "input.cl", "input", NULL, &rt->iprg, &rt->iknl);
	}

	if(0 == r)
	{
		inm = rte_cl_create_buffer(nn, NHWC_SIZE(*nhwc), in);

		if(NULL != inm)
		{
			r = cl_set_kernel_args(rt->iknl, RTE_CL_ARGS_WITH_NHWC, nhwc, 2,
							sizeof(cl_mem), &inm,
							sizeof(cl_mem), &img2d);

			if(0 == r)
			{
				r = cl_enqueue_kernel(nn, rt->iknl, nhwc, RTE_GWT_W_H_C, TRUE);
			}

			clReleaseMemObject(inm);
		}
		else
		{
			r = NN_E_NO_MEMORY;
		}
	}

	return r;
}

int rte_cl_image2d_copy_out(const nn_t* nn, cl_mem img2d, float* out, NHWC_t* nhwc)
{
	int r = 0;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;
	cl_program program;
	cl_kernel kernel;
	cl_mem outm;

	if(NULL == rt->oknl)
	{
		r = cl_create_kernel(nn, OPENCL_PATH "output.cl", "output", NULL, &rt->oprg, &rt->oknl);
	}

	if(0 == r)
	{
		outm = rte_cl_create_buffer(nn, NHWC_SIZE(*nhwc), NULL);

		if(NULL != outm)
		{
			r = cl_set_kernel_args(rt->oknl, RTE_CL_ARGS_WITH_NHWC, nhwc, 2,
							sizeof(cl_mem), &img2d,
							sizeof(cl_mem), &outm);

			if(0 == r)
			{
				r = cl_enqueue_kernel(nn, rt->oknl, nhwc, RTE_GWT_W_H_C, TRUE);
			}

			if(0 == r)
			{
				r = rte_cl_read_buffer(nn, outm, out, NHWC_SIZE(*nhwc));
			}

			clReleaseMemObject(outm);
		}
		else
		{
			r = NN_E_NO_MEMORY;
		}
	}

	return r;
}

cl_mem rte_cl_create_image2d_from_blob(const nn_t* nn, const layer_blob_t* blob)
{
	cl_mem img2d = NULL;
	int r = 0;
	NHWC_t nhwc;

	r = layer_get_blob_NHWC(blob, &nhwc);

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
			const char* option,
			size_t sz, size_t nout)
{
	int r = 0;
	cl_int errNum;
	layer_cl_context_t* context = NULL;

	assert(sz >= sizeof(layer_cl_context_t));

	context = malloc(sz+nout*sizeof(cl_mem));

	if(context != NULL)
	{
		memset(context, 0, sz);
		context->dtype = L_DT_FLOAT;
		context->out = (void*)(((unsigned long long)context)+sz);
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
		if(program != NULL)
		{
			r = cl_create_kernel(nn, program, kernel, option, &context->program, &context->kernel);
		}
		else
		{
			context->program = NULL;
			context->kernel = NULL;
		}
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
#ifndef ENABLE_CL_IMAGE_REUSE
	size_t i;
#endif
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;

	if(NULL != context)
	{
		if(context->kernel != NULL)
		{
			clReleaseKernel(context->kernel);
		}

		if(context->program != NULL)
		{
			clReleaseProgram(context->program);
		}
#ifndef ENABLE_CL_IMAGE_REUSE
		for(i=0; i<context->nout; i++)
		{
			if(NULL != context->out[i])
			{
				clReleaseMemObject(context->out[i]);
			}
		}
#else
		if(layer->op == L_OP_OUTPUT)
		{	/* output only has one cl buffer object not managed by rt->images */
			if(NULL != context->out[0])
			{
				clReleaseMemObject(context->out[0]);
			}
		}
#endif
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

int rte_cl_execute_layer(const nn_t* nn, const layer_t* layer, rte_cl_global_work_type_t gwt, int run, NHWC_t* nhwc)
{
	int r = 0;
	layer_cl_context_t* context = (layer_cl_context_t*)layer->C->context;

	if(NULL == nhwc)
	{
		nhwc = &(context->nhwc);
	}

	r = cl_enqueue_kernel(nn, context->kernel, nhwc, gwt, run);

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
#ifdef ENABLE_CL_IMAGE_REUSE
void* rte_cl_alloc_image2d(const nn_t* nn, const layer_t* layer, int H, int W)
{
	int r;
	rte_cl_image_t* image = NULL;
	rte_cl_image_t* i;
	rte_cl_t* rt = (rte_cl_t*)nn->runtime;

	STAILQ_FOREACH(i, &(rt->images), entry)
	{
		if(NULL == i->owner)
		{
			image = i;
			break;
		}
		else
		{
			r = rte_is_layer_consumed_from(nn, i->owner, layer);
			if(FALSE == r)
			{
				image = i;
				break;
			}
		}
	}

	if(NULL == image)
	{
		image = malloc(sizeof(rte_cl_image_t));
		if(NULL != image)
		{
			image->owner = layer;
			image->H = H;
			image->W = W;
			image->img = NULL;

			STAILQ_INSERT_TAIL(&(rt->images), image, entry);
		}
	}
	else
	{
		image->owner = layer;
		if(H > image->H)
		{
			image->H = H;
		}
		if(W > image->W)
		{
			image->W = W;
		}
	}

	return image;
}
#endif /* ENABLE_CL_IMAGE_REUSE */

int rte_cl_create_layer_common(const nn_t* nn, const layer_t* layer,
		const char* program, const char* kernel, const char* option, size_t ctx_sz)
{
	int r = 0;

	layer_cl_context_t* context;

	r = rte_cl_create_layer_context(nn, layer,
				program, kernel, option, ctx_sz, 1);

	if(0 == r)
	{
		context = (layer_cl_context_t*)layer->C->context;

		RTE_CL_LOG_LAYER_SHAPE(layer);
#ifdef ENABLE_CL_IMAGE_REUSE
		context->out[0] = (cl_mem)rte_cl_alloc_image2d(nn, layer,
#else
		context->out[0] = (cl_mem)rte_cl_create_image2d(nn,
#endif
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

#ifndef DISABLE_RTE_FALLBACK
extern void rte_cpuq_to_cpu_float_init_common(const nn_t* nn, const layer_t* layer);
extern void rte_cpuq_to_cpu_float_post_execute_common(const nn_t* nn, const layer_t* layer);
void rte_cl_to_cpu_float_init_common(const nn_t* nn, const layer_t* layer)
{
	if(L_OP_YOLOOUTPUT != layer->op)
	{
		rte_cpuq_to_cpu_float_init_common(nn, layer);
	}

}
int rte_cl_to_cpu_float_pre_execute_common(const nn_t* nn, const layer_t* layer)
{
	int r=0;
	layer_cl_context_t* context;
	const layer_t* const* inputs;
	void** cl_inputs = (void**)nn->scratch.area;
	float* pf;

	if(L_OP_YOLOOUTPUT == layer->op)
	{
		return 0;
	}

	inputs = layer->inputs;
	while(NULL != (*inputs))
	{
		context = (layer_cl_context_t*)(*inputs)->C->context;
		*cl_inputs++ = context->out[0];
		inputs++;
	}

	pf = (float*)cl_inputs;

	inputs = layer->inputs;
	while((NULL != (*inputs)) && (0 == r))
	{
		context = (layer_cl_context_t*)(*inputs)->C->context;
		r = rte_cl_image2d_copy_out(nn, context->out[0], pf, &(context->nhwc));
		context->out[0] = pf;
		pf += NHWC_SIZE(context->nhwc);
		inputs++;
	}

	return r;
}
void rte_cl_to_cpu_float_post_execute_common(const nn_t* nn, const layer_t* layer)
{
	if(L_OP_YOLOOUTPUT != layer->op)
	{
		rte_cpuq_to_cpu_float_post_execute_common(nn, layer);
	}

}
#endif /* DISABLE_RTE_FALLBACK */
#endif /* DISABLE_RUNTIME_OPENCL */
