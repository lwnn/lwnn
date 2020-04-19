/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#include "image.h"
#include <vector>
#include <cmath>
#include <cstdlib>
#include <stdio.h>
#include <assert.h>
#include "bbox_util.hpp"
/* ============================ [ MACROS    ] ====================================================== */
#define FACE_MIN_SIZE 20

#define PNet_PROPOSAL_INPUTS NULL
#define PNet_PROPOSAL_DIMS 10,4
#define l_blobs_PNet_PROPOSAL NULL
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
L_PROPOSAL (PNet_PROPOSAL, PNet_PROPOSAL_INPUTS);
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
int mtcnn_predict(nn_t* PNet, nn_t* RNet, nn_t* ONet, image_t* im, float** p_points, size_t* p_number)
{
	int r = 0;
	size_t number = 0;
	float* points = NULL;
	nn_input_t* PNet_In = (nn_input_t*)PNet->network->inputs[0];

	const layer_t* RPL = &l_layer_PNet_PROPOSAL;
	RPL->C->context = new layer_context_t;
	if(NULL != RPL->C->context) {
		r = layer_get_NHWC(RPL, &(RPL->C->context->nhwc));
	} else {
		r = NN_E_NO_MEMORY;
	}

	const float factor = 0.709;
	const float threshold[] = { 0.6, 0.7, 0.7 };

	size_t factor_count=0;
	float h=im->h;
	float w=im->w;
	float minl=std::fmin(h, w);
	float m = 12.0/FACE_MIN_SIZE;
	minl=minl*m;
	/* create scale pyramid */
	std::vector<float> scales;
	while((minl>=12) && (0==r)) {
		scales.push_back(m*std::pow(factor, factor_count));
		minl = minl*factor;
		factor_count += 1;
	}

	NNLOG(NN_DEBUG, ("MTCNN %d scales for image [%dx%d]\n", (int)scales.size(), im->h, im->w));
	for(int i=0; (i<scales.size()) && (0==r); i++) {
		float scale = scales[i];
		int hs=int(std::ceil(h*scale));
		int ws=int(std::ceil(w*scale));
		NNLOG(NN_DEBUG, ("sampled image [%dx%d] by scale %.3f\n", hs, ws, scale));
		image_t* im_data = image_resize(im, ws, hs);
		float* img_y = new float[ws*hs*im->c]; /* in format NWHC */
		if((NULL!=img_y) && (NULL!=im_data)) {
			for(int y=0; y<hs; y++) {
				for(int x=0; x<ws; x++) {
					for(int z=0; z<im->c; z++) {
						img_y[(x*hs+y)*im->c+z] = (im_data->data[(y*ws+x)*im->c+z]-127.5)*0.0078125;
					}
				}
			}
			PNet_In->data = img_y;
			PNet_In->layer->C->context->nhwc.H = ws;
			PNet_In->layer->C->context->nhwc.W = hs;
			r = nn_predict(PNet);
		} else {
			r = NN_E_NO_MEMORY;
		}
		if(im_data) image_close(im_data);
		if(img_y) delete [] img_y;
	}

	*p_number = number;
	*p_points = points;
	return r;
}
