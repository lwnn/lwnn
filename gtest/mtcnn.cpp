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

#define ANCHOR_STRIDE    2
#define ANCHOR_CELL_SIZE 12

#define PNET_TOPK 100

#define PNet_PROPOSAL_INPUTS NULL
#define PNet_PROPOSAL_DIMS 1,PNET_TOPK,4
#define l_blobs_PNet_PROPOSAL NULL
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
L_PROPOSAL (PNet_PROPOSAL, PNet_PROPOSAL_INPUTS);
/* ============================ [ LOCALS    ] ====================================================== */
static float* generate_anchors(float* scores, int H, int W, float scale, int iH, int iW) {

	float* anchors = new float[H*W*4];
	int count = 0;
	if(anchors != NULL) {
		float* bbox = anchors;
		for(int h=0;h<H;h++){
			for(int w=0;w<W;w++){
				bbox[0] = std::round((ANCHOR_STRIDE*h+1)/scale)/iH;
				bbox[1] = std::round((ANCHOR_STRIDE*w+1)/scale)/iW;
				bbox[2] = std::round((ANCHOR_STRIDE*h+1+ANCHOR_CELL_SIZE)/scale)/iH;
				bbox[3] = std::round((ANCHOR_STRIDE*w+1+ANCHOR_CELL_SIZE)/scale)/iW;
				bbox += 4;
			}
		}
	}
	return anchors;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
/* ref https://github.com/AlphaQi/MTCNN-light/blob/master/src/mtcnn.cpp */
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

	float* top_data = new float[PNET_TOPK*4];
	if(NULL == top_data) {
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

	NNLOG(NN_DEBUG, ("MTCNN %d scales for image [%dx%dx%d]\n", (int)scales.size(), im->h, im->w, im->c));
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
			if(0 == r) {
				float* scores = (float*)PNet->network->outputs[0]->layer->C->context->out[0];
				float* locations = (float*)PNet->network->outputs[1]->layer->C->context->out[0];
				int W = PNet->network->outputs[0]->layer->C->context->nhwc.H;
				int H = PNet->network->outputs[0]->layer->C->context->nhwc.W;
				float* anchors = generate_anchors(scores, H, W, scale, im->h, im->w);
				const float var_data[4] = { 1, 1, 1, 1 };

				if(anchors != NULL) {
					NNLOG(NN_DEBUG, ("  prob: %dx%dx%dx%d, bbox: %dx%dx%dx%d\n",
											L_SHAPES(PNet->network->outputs[0]->layer),
											L_SHAPES(PNet->network->outputs[1]->layer)));
					layer_get_NHWC(RPL, &RPL->C->context->nhwc);
					r = ssd::detection_output_forward(locations, scores, anchors,
							(const float*)&var_data, top_data, H*W, 1, 0.5, threshold[0], 2, true,
							0, PNET_TOPK, PNET_TOPK, ssd::PriorBoxParameter_CodeType_CORNER_SIZE,
							true, 1.0, RPL);
					delete [] anchors;
				} else {
					r = NN_E_NO_MEMORY;
				}
				r = -999;
			}
		} else {
			r = NN_E_NO_MEMORY;
		}
		if(im_data) image_close(im_data);
		if(img_y) delete [] img_y;
	}

	if(top_data) delete [] top_data;
	if(RPL->C->context) delete RPL->C->context;

	*p_number = number;
	*p_points = points;
	return r;
}
