/**
 * NNCL - Neural Network on openCL
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include <stdlib.h>
#include <stdio.h>
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
struct nn {
	cl_context context;
};
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
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
		printf("Failed to find any OpenCL platforms.\n");
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
/* ============================ [ FUNCTIONS ] ====================================================== */
nn_t* nn_create(void)
{
	nn_t* nn;

	nn = malloc(sizeof(nn_t));
	if(NULL != nn)
	{
		nn->context = nn_create_context();
	}

	if(NULL == nn->context)
	{
		free(nn);
		nn = NULL;
	}

	return nn;
}

