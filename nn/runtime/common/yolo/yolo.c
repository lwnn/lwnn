/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#if !defined(DISABLE_RUNTIME_CPU_FLOAT) || !defined(DISABLE_RUNTIME_OPENCL)
#include "yolo.h"
#include "algorithm.h"
#include <math.h>
#include <stdlib.h>
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct{
    float x, y, w, h;
} box;

typedef struct detection{
    box bbox;
    int classes;
    float *prob;
    float *mask;
    float objectness;
    int sort_class;
} detection;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static inline float logistic_activate(float x){return 1./(1. + exp(-x));}
static void activate_array(float *x, const int n)
{
	int i;
	for(i = 0; i < n; ++i){
		x[i] = logistic_activate(x[i]);
	}
}

static int entry_index(NHWC_t *inhwc, int classes, int batch, int location, int entry)
{
	int n =   location / (inhwc->W*inhwc->H);
	int loc = location % (inhwc->W*inhwc->H);
	return batch*NHWC_BATCH_SIZE(*inhwc) + n*inhwc->W*inhwc->H*(4+classes+1) + entry*inhwc->W*inhwc->H + loc;
}

static int yolo_num_detections(const nn_t* nn, const layer_t* layer, float thresh)
{
	int i, n;
	float* output = (float*)layer->C->context->out[0];
	int count = 0;
	int num = layer->blobs[0]->dims[0];
	int classes = RTE_FETCH_FLOAT(layer->blobs[2]->blob, 0);
	layer_context_t* context = (layer_context_t*)layer->C->context;

	for (i = 0; i < context->nhwc.W*context->nhwc.H; ++i){
		for(n = 0; n < num; ++n){
			int obj_index  = entry_index(&context->nhwc, classes, 0,
					n*context->nhwc.W*context->nhwc.H + i, 4);
			if(output[obj_index] > thresh){
				++count;
			}
		}
	}

	return count;
}
static int num_detections(const nn_t* nn, const layer_t* layer, float thresh)
{
	const layer_t ** inputs = layer->inputs;
	int s = 0;

	while((*inputs) != NULL) {
		s += yolo_num_detections(nn, *inputs, thresh);
		inputs++;
	}
	return s;
}

static detection *make_network_boxes(const nn_t* nn, const layer_t* layer, float thresh, int *num)
{
	int i;
	const layer_t* input = layer->inputs[0];
	int classes = RTE_FETCH_FLOAT(input->blobs[2]->blob, 0);
	int coords = 0; /* TODO: what is coords */
	int nboxes = num_detections(nn, layer, thresh);
	NNLOG(NN_DEBUG, ("  detected boxes number = %d\n", nboxes));
	if(num) *num = nboxes;
	detection *dets = calloc(nboxes, sizeof(detection));
	for(i = 0; i < nboxes; ++i){
		dets[i].prob = calloc(classes, sizeof(float));
		if(coords > 4){
			dets[i].mask = calloc(coords-4, sizeof(float));
		}
	}
	return dets;
}

static void avg_flipped_yolo(const nn_t* nn, const layer_t* layer)
{
	int i,j,n,z;
	layer_context_t* context = (layer_context_t*)layer->C->context;
	float *output = (float*)layer->C->context->out[0];
	int num = layer->blobs[0]->dims[0];
	int classes = RTE_FETCH_FLOAT(layer->blobs[2]->blob, 0);
	float *flip = output + NHWC_BATCH_SIZE(context->nhwc);
	for (j = 0; j < context->nhwc.H; ++j) {
		for (i = 0; i < context->nhwc.W/2; ++i) {
			for (n = 0; n < num; ++n) {
				for(z = 0; z < classes + 4 + 1; ++z){
					int i1 = z*context->nhwc.W*context->nhwc.H*num + n*context->nhwc.W*context->nhwc.H + j*context->nhwc.W + i;
					int i2 = z*context->nhwc.W*context->nhwc.H*num + n*context->nhwc.W*context->nhwc.H + j*context->nhwc.W + (context->nhwc.W - i - 1);
					float swap = flip[i1];
					flip[i1] = flip[i2];
					flip[i2] = swap;
					if(z == 0){
						flip[i1] = -flip[i1];
						flip[i2] = -flip[i2];
					}
				}
			}
		}
	}
	for(i = 0; i < NHWC_BATCH_SIZE(context->nhwc); ++i){
		output[i] = (output[i] + flip[i])/2.;
	}
}

static box get_yolo_box(float *x, const int *anchors, int n, int index, int i, int j, int lw, int lh, int w, int h, int stride)
{
	box b;
	b.x = (i + x[index + 0*stride]) / lw;
	b.y = (j + x[index + 1*stride]) / lh;
	b.w = exp(x[index + 2*stride]) * anchors[2*n]   / w;
	b.h = exp(x[index + 3*stride]) * anchors[2*n+1] / h;
	return b;
}

static int get_yolo_detections(const nn_t* nn, const layer_t* layer,
			float thresh, int *map, int relative, detection *dets)
{
	int i,j,n;
	float *predictions = (float*)layer->C->context->out[0];
	int num = layer->blobs[0]->dims[0];
	const int* mask = (const int*)layer->blobs[0]->blob;
	const int* anchors = (const int*)layer->blobs[1]->blob;
	int classes = RTE_FETCH_FLOAT(layer->blobs[2]->blob, 0);
	layer_context_t* context = (layer_context_t*)layer->C->context;
	if (context->nhwc.N == 2) avg_flipped_yolo(nn, layer);
	int count = 0;
	layer_context_t* image_context = (layer_context_t*)nn->network->inputs[0]->layer->C->context;
	int netw = image_context->nhwc.W;
	int neth = image_context->nhwc.H;

	for (i = 0; i < context->nhwc.W*context->nhwc.H; ++i){
		int row = i / context->nhwc.W;
		int col = i % context->nhwc.W;
		for(n = 0; n < num; ++n){
			int obj_index  = entry_index(&context->nhwc, classes, 0, n*context->nhwc.W*context->nhwc.H + i, 4);
			float objectness = predictions[obj_index];
			if(objectness <= thresh) continue;
			int box_index  = entry_index(&context->nhwc, classes, 0, n*context->nhwc.W*context->nhwc.H + i, 0);
			dets[count].bbox = get_yolo_box(predictions, anchors, mask[n], box_index, col, row, context->nhwc.W, context->nhwc.H, netw, neth, context->nhwc.W*context->nhwc.H);
			dets[count].objectness = objectness;
			dets[count].classes = classes;
			for(j = 0; j < classes; ++j){
				int class_index = entry_index(&context->nhwc, classes, 0, n*context->nhwc.W*context->nhwc.H + i, 4 + 1 + j);
				float prob = objectness*predictions[class_index];
				dets[count].prob[j] = (prob > thresh) ? prob : 0;
			}
			++count;
		}
	}

	return count;
}

static void fill_network_boxes(const nn_t* nn, const layer_t* layer,
			float thresh, float hier, int *map, int relative, detection *dets)
{
	const layer_t ** inputs = layer->inputs;

	while((*inputs) != NULL) {
		int count = get_yolo_detections(nn, *inputs, thresh, map, relative, dets);
		dets += count;
		inputs++;
	}
}

static detection *get_network_boxes(const nn_t* nn, const layer_t* layer,
			float thresh, float hier, int *map, int relative, int *num)
{
	detection *dets = make_network_boxes(nn, layer, thresh, num);
	fill_network_boxes(nn, layer, thresh, hier, map, relative, dets);
	return dets;
}

static int nms_comparator(const void *pa, const void *pb)
{
	detection a = *(detection *)pa;
	detection b = *(detection *)pb;
	float diff = 0;
	if(b.sort_class >= 0){
		diff = a.prob[b.sort_class] - b.prob[b.sort_class];
	} else {
		diff = a.objectness - b.objectness;
	}
	if(diff < 0) return 1;
	else if(diff > 0) return -1;
	return 0;
}


static float overlap(float x1, float w1, float x2, float w2)
{
	float l1 = x1 - w1/2;
	float l2 = x2 - w2/2;
	float left = l1 > l2 ? l1 : l2;
	float r1 = x1 + w1/2;
	float r2 = x2 + w2/2;
	float right = r1 < r2 ? r1 : r2;
	return right - left;
}

static float box_intersection(box a, box b)
{
	float w = overlap(a.x, a.w, b.x, b.w);
	float h = overlap(a.y, a.h, b.y, b.h);
	if(w < 0 || h < 0) return 0;
	float area = w*h;
	return area;
}

static float box_union(box a, box b)
{
	float i = box_intersection(a, b);
	float u = a.w*a.h + b.w*b.h - i;
	return u;
}

static float box_iou(box a, box b)
{
	return box_intersection(a, b)/box_union(a, b);
}

static void do_nms_sort(detection *dets, int total, int classes, float thresh)
{
	int i, j, k;

	NNLOG(NN_DEBUG, ("  do nms sort(%d, %d, %.2f)\n", total, classes, thresh));
	k = total-1;
	for(i = 0; i <= k; ++i){
		if(dets[i].objectness == 0){
			detection swap = dets[i];
			dets[i] = dets[k];
			dets[k] = swap;
			--k;
			--i;
		}
	}
	total = k+1;

	for(k = 0; k < classes; ++k){
		for(i = 0; i < total; ++i){
			dets[i].sort_class = k;
		}
		qsort(dets, total, sizeof(detection), nms_comparator);
		for(i = 0; i < total; ++i){
			if(dets[i].prob[k] == 0) continue;
			box a = dets[i].bbox;
			for(j = i+1; j < total; ++j){
				box b = dets[j].bbox;
				if (box_iou(a, b) > thresh){
					dets[j].prob[k] = 0;
				}
			}
		}
	}
}

static int save_detections(float* output, detection *dets, int num, float thresh, int classes)
{
	int i,j;
	int class;
	int count = 0;
	float scores = 0;

	for(i = 0; i < num; ++i){
		class = -1;
		for(j = 0; j < classes; ++j){
			if (dets[i].prob[j] > thresh){
				if (class < 0) {
					scores = dets[i].prob[j];
					class = j;
				} else if(dets[i].prob[j] > scores) {
					scores = dets[i].prob[j];
					class = j;
				} else {
					/* do nothing */
				}
			}
		}

		if(class >= 0){
			box b = dets[i].bbox;

			output[count * 7] = 0;
			output[count * 7 + 1] = class;
			output[count * 7 + 2] = scores;
			output[count * 7 + 3] = b.x;
			output[count * 7 + 4] = b.y;
			output[count * 7 + 5] = b.w;
			output[count * 7 + 6] = b.h;
			++count;
			NNLOG(NN_DEBUG, (" detect B=0 L=%d P=%.2f @ [%.2f %.2f %.2f %.2f]\n",
					class, scores,
					b.x, b.y, b.w, b.h));
		}
	}

	return count;
}

static void free_detections(detection *dets, int n)
{
	int i;
	for(i = 0; i < n; ++i){
		free(dets[i].prob);
		if(dets[i].mask) free(dets[i].mask);
	}
	free(dets);
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int yolo_forward(float* output, const float* input, NHWC_t *inhwc, int num, int classes)
{
	int r = 0;
	int b,n;

	r = alg_transpose(output, input, inhwc, sizeof(float), ALG_TRANSPOSE_FROM_NHWC_TO_NCHW);

	for (b = 0; b < inhwc->N; ++b) {
		for(n = 0; n < num; ++n){
			int index = entry_index(inhwc, classes, b, n*inhwc->W*inhwc->H, 0);
			activate_array(output + index, 2*inhwc->W*inhwc->H);
			index = entry_index(inhwc, classes, b, n*inhwc->W*inhwc->H, 4);
			activate_array(output + index, (1+classes)*inhwc->W*inhwc->H);
		}
	}

	return r;
}

int yolo_output_forward(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_context_t* context = (layer_context_t*)layer->C->context;
	const layer_t* input = layer->inputs[0];
	int classes = RTE_FETCH_FLOAT(input->blobs[2]->blob, 0);
	int nboxes = 0;
	float* output = (float*)nn_get_output_data(nn, layer);

	float thresh = 0.5;
	float hier_thresh = 0.5;
	float nms = 0.45;

	detection *dets = get_network_boxes(nn, layer, thresh, hier_thresh, 0, 1, &nboxes);
	do_nms_sort(dets, nboxes, classes, 0.45);

	layer_get_NHWC(layer, &context->nhwc);
	if(nboxes > (context->nhwc.N*context->nhwc.H)) {
		nboxes = (context->nhwc.N*context->nhwc.H);
	}
	context->nhwc.H = 1;
	context->nhwc.N = save_detections(output, dets, nboxes, thresh, classes);
	free_detections(dets, nboxes);

	return r;
}
#endif /* DISABLE_RUNTIME_CPU_FLOAT/DISABLE_RUNTIME_OPENCL */
