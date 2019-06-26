/**
 * NNCL - Neural Network on openCL
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include "layer.h"
#include <stdlib.h>
#include <stdio.h>
#include <CL/cl.h>
/* ============================ [ MACROS    ] ====================================================== */

/* ============================ [ TYPES     ] ====================================================== */
struct nn {
	cl_context context;
	cl_device_id device;
	cl_command_queue command_queue;

	const layer_t* const* network;
};
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
int nn_log_level = 1;
/* ============================ [ LOCALS    ] ====================================================== */
static cl_context nn_create_context()
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

static cl_command_queue nn_create_command_queue(cl_context context, cl_device_id *device)
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
static int nn_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	NN_LOG(NN_DEBUG, ("  run %-16s: op=%d\n", layer->name, layer->op));
	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
nn_t* nn_create(const layer_t* const* network)
{
	nn_t* nn;

	nn = malloc(sizeof(nn_t));
	if(NULL != nn)
	{
		nn->context = nn_create_context();
		nn->network = network;
	}

	if(NULL == nn->context)
	{
		free(nn);
		nn = NULL;
	}
	else
	{
		nn->command_queue = nn_create_command_queue(nn->context, &nn->device);
	}

	if(NULL == nn->command_queue)
	{
		free(nn->context);
		free(nn);
		nn = NULL;
	}

	return nn;
}

void nn_set_log_level(int level)
{
	nn_log_level = level;
}

int nn_predict(const nn_t* nn)
{
	int r = 0;
	const layer_t* const* network;
	const layer_t* layer;

	network = nn->network;

	layer = *network++;
	while((NULL != layer) && (0 == r))
	{
		r = nn_execute_layer(nn, layer);
		layer = *network++;
	}

	return r;
}

