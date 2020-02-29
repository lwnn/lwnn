/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn_test_util.h"
#include "bbox_util.hpp"
#include "image.h"
#include <chrono>
/* ============================ [ MACROS    ] ====================================================== */
#define NNT_MNIST_NOT_FOUND_OKAY FALSE
#define NNT_MNIST_TOP1 0.9

#define NNT_UCI_INCEPTION_NOT_FOUND_OKAY TRUE
#define NNT_UCI_INCEPTION_TOP1 0.85

#define NNT_SSD_NOT_FOUND_OKAY TRUE
#define NNT_SSD_TOP1 0.9

#define NNT_YOLOV3_NOT_FOUND_OKAY TRUE
#define NNT_YOLOV3_TOP1 0.9

#define NNT_YOLOV3TINY_NOT_FOUND_OKAY TRUE
#define NNT_YOLOV3TINY_TOP1 0.9

#define NNT_VEHICLE_ATTR_NOT_FOUND_OKAY TRUE
#define NNT_VEHICLE_ATTR_TOP1 0.9

#define NNT_ENET_NOT_FOUND_OKAY TRUE
#define NNT_ENET_TOP1 0.9

#define NNT_KWS_NOT_FOUND_OKAY TRUE
#define NNT_KWS_TOP1 0.9

#define NNT_DS_NOT_FOUND_OKAY TRUE
#define NNT_DS_TOP1 0.9

#define ARGS_PREFETCH_ALL 0x01
#define ARGS_COMPARE_ALL_AT_END 0x02
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	void* (*load_input)(nn_t* nn, const char* path, int id, size_t* sz);
	void* (*load_output)(const char* path, int id, size_t* sz);
	int (*compare)(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg);
	size_t n;
	size_t flags;
} nnt_model_args_t;
/* ============================ [ DECLARES  ] ====================================================== */
static void* load_input(nn_t* nn, const char* path, int id, size_t* sz);
static void* load_ssd_input(nn_t* nn, const char* path, int id, size_t* sz);
static void* load_yolov3_input(nn_t* nn, const char* path, int id, size_t* sz);
static void* load_vehicle_attributes_recognition_barrier_0039_input(nn_t* nn, const char* path, int id, size_t* sz);
static void* load_enet_input(nn_t* nn, const char* path, int id, size_t* sz);
static void* load_kws_input(nn_t* nn, const char* path, int id, size_t* sz);
static void* load_ds_input(nn_t* nn, const char* path, int id, size_t* sz);
static void* load_output(const char* path, int id, size_t* sz);
static int ssd_compare(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg);
static int yolov3_compare(nn_t* nn, int id, float* output, size_t szo, float* gloden, size_t szg);
static int vehicle_attributes_recognition_barrier_0039_compare(nn_t* nn, int id, float* output, size_t szo, float* gloden, size_t szg);
static int enet_compare(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg);
static int kws_compare(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg);
static int ds_compare(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg);
/* ============================ [ DATAS     ] ====================================================== */
const char* g_InputImagePath = NULL;

static const char* voc_names_for_ssd[] = {"background","aeroplane", "bicycle", "bird", "boat","bottle", "bus", "car", "cat", "chair","cow", "diningtable", "dog", "horse","motorbike", "person", "pottedplant","sheep", "sofa", "train", "tvmonitor"};
static const char* coco_names_for_yolo[] = {"person", "bicycle", "car", "motorbike", "aeroplane", "bus", "train", "truck","boat", "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "sofa", "pottedplant", "bed", "diningtable", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush" };

NNT_CASE_DEF(MNIST) =
{
	NNT_CASE_DESC(mnist),
};

NNT_CASE_DEF(UCI_INCEPTION) =
{
	NNT_CASE_DESC(uci_inception),
};

static const nnt_model_args_t nnt_ssd_args =
{
	load_ssd_input,
	load_output,
	ssd_compare,
	7	/* 7 test images */
};

NNT_CASE_DEF(SSD) =
{
	NNT_CASE_DESC_ARGS(ssd),
};

static const nnt_model_args_t nnt_yolov3_args =
{
	load_yolov3_input,
	load_output,
	yolov3_compare,
	1	/* 1 test images */
};

static const nnt_model_args_t nnt_yolov3_tiny_args =
{
	load_yolov3_input,
	load_output,
	yolov3_compare,
	1	/* 1 test images */
};

NNT_CASE_DEF(YOLOV3) =
{
	NNT_CASE_DESC_ARGS(yolov3),
};

NNT_CASE_DEF(YOLOV3TINY) =
{
	NNT_CASE_DESC_ARGS(yolov3_tiny),
};

static const nnt_model_args_t nnt_vehicle_attributes_recognition_barrier_0039_args =
{
	load_vehicle_attributes_recognition_barrier_0039_input,
	load_output,
	vehicle_attributes_recognition_barrier_0039_compare,
	1
};

NNT_CASE_DEF(VEHICLE_ATTR) =
{
	NNT_CASE_DESC_ARGS(vehicle_attributes_recognition_barrier_0039),
};

static const nnt_model_args_t nnt_enet_args =
{
	load_enet_input,
	load_output,
	enet_compare,
	1
};

NNT_CASE_DEF(ENET) =
{
	NNT_CASE_DESC_ARGS(enet),
};

static const nnt_model_args_t nnt_kws_args =
{
	load_kws_input,
	load_output,
	kws_compare,
	1
};

NNT_CASE_DEF(KWS) =
{
	NNT_CASE_DESC_ARGS(kws),
};

static const nnt_model_args_t nnt_deepspeech_args =
{
	load_ds_input,
	load_output,
	ds_compare,
	1,
	ARGS_PREFETCH_ALL|ARGS_COMPARE_ALL_AT_END
};

NNT_CASE_DEF(DS) =
{
	NNT_CASE_DESC_ARGS(deepspeech),
};
/* ============================ [ LOCALS    ] ====================================================== */
static void* load_input(nn_t* nn, const char* path, int id, size_t* sz)
{
	char name[256];
	snprintf(name, sizeof(name), "%s/input%d.raw", path, id);

	return nnt_load(name, sz);
}

static void* load_ssd_input(nn_t* nn, const char* path, int id, size_t* sz)
{
	image_t* im;
	image_t* resized_im;
	layer_context_t* context = (layer_context_t*)nn->network->inputs[0]->layer->C->context;

	EXPECT_EQ(context->nhwc.C, 3);

	if(g_InputImagePath != NULL)
	{
		printf("loading %s for %s\n", g_InputImagePath, nn->network->name);
		im = image_open(g_InputImagePath);
		assert(im != NULL);
		resized_im = image_resize(im, context->nhwc.W, context->nhwc.H);
		assert(resized_im != NULL);
		float* input = (float*)malloc(sizeof(float)*NHWC_BATCH_SIZE(context->nhwc));

		for(int i=0; i<NHWC_BATCH_SIZE(context->nhwc); i++)
		{
			input[i] = 0.007843*(resized_im->data[i]-127.5);
		}

		image_close(im);
		image_close(resized_im);

		*sz = sizeof(float)*NHWC_BATCH_SIZE(context->nhwc);
		return (void*) input;
	}
	else
	{
		return load_input(nn, path, id, sz);
	}
}

static void* load_yolov3_input(nn_t* nn, const char* path, int id, size_t* sz)
{
	image_t* im;
	image_t* resized_im;
	layer_context_t* context = (layer_context_t*)nn->network->inputs[0]->layer->C->context;

	EXPECT_EQ(context->nhwc.C, 3);

	if(g_InputImagePath != NULL)
	{
		printf("loading %s for %s\n", g_InputImagePath, nn->network->name);
		im = image_open(g_InputImagePath);
		assert(im != NULL);
		resized_im = image_letterbox(im, context->nhwc.W, context->nhwc.H);
		assert(resized_im != NULL);
		float* input = (float*)malloc(sizeof(float)*NHWC_BATCH_SIZE(context->nhwc));

		for(int i=0; i<NHWC_BATCH_SIZE(context->nhwc); i++)
		{
			input[i] = resized_im->data[i]/255.f;
		}
		image_close(im);
		image_close(resized_im);

		*sz = sizeof(float)*NHWC_BATCH_SIZE(context->nhwc);
		return (void*) input;
	}
	else
	{
		return load_input(nn, path, id, sz);
	}
}

static void* load_vehicle_attributes_recognition_barrier_0039_input(nn_t* nn, const char* path, int id, size_t* sz)
{
	image_t* im;
	image_t* resized_im;
	layer_context_t* context = (layer_context_t*)nn->network->inputs[0]->layer->C->context;

	EXPECT_EQ(context->nhwc.C, 3);

	assert(g_InputImagePath != NULL);

	printf("loading %s for %s\n", g_InputImagePath, nn->network->name);
	im = image_open(g_InputImagePath);
	assert(im != NULL);
	resized_im = image_resize(im, context->nhwc.W, context->nhwc.H);
	assert(resized_im != NULL);
	float* input = (float*)malloc(sizeof(float)*NHWC_BATCH_SIZE(context->nhwc));

	for(int i=0; i<NHWC_BATCH_SIZE(context->nhwc); i++)
	{
		input[i] = resized_im->data[i];
	}
	image_close(im);
	image_close(resized_im);

	*sz = sizeof(float)*NHWC_BATCH_SIZE(context->nhwc);
	return (void*) input;
}

static void* load_enet_input(nn_t* nn, const char* path, int id, size_t* sz)
{
	image_t* im;
	image_t* resized_im;
	layer_context_t* context = (layer_context_t*)nn->network->inputs[0]->layer->C->context;

	EXPECT_EQ(context->nhwc.C, 3);

	if(g_InputImagePath != NULL)
	{
		printf("loading %s for %s\n", g_InputImagePath, nn->network->name);
		im = image_open(g_InputImagePath);
		assert(im != NULL);
		resized_im = image_resize(im, context->nhwc.W, context->nhwc.H);
		assert(resized_im != NULL);
		float* input = (float*)malloc(sizeof(float)*NHWC_BATCH_SIZE(context->nhwc));

		for(int i=0; i<NHWC_BATCH_SIZE(context->nhwc)/3; i++)
		{	/* BGR */
			input[3*i] = resized_im->data[3*i+2];
			input[3*i+1] = resized_im->data[3*i+1];
			input[3*i+2] = resized_im->data[3*i];
		}
		image_close(im);
		image_close(resized_im);

		*sz = sizeof(float)*NHWC_BATCH_SIZE(context->nhwc);
		return (void*) input;
	}
	else
	{
		return load_input(nn, path, id, sz);
	}
}

static void* load_kws_input(nn_t* nn, const char* path, int id, size_t* sz)
{
	assert(g_InputImagePath != NULL);

	void* wav_data = nnt_load(g_InputImagePath, sz);

	void** inputs = (void**)malloc(sizeof(void*)*3);
	inputs[0] = wav_data;
	inputs[1] = (void*)*sz;

	printf("load %s @%p with size %d\n", g_InputImagePath, wav_data, (int)*sz);

	return inputs;
}

static void* load_ds_input(nn_t* nn, const char* path, int id, size_t* sz)
{
	return NULL;
}

static void* load_output(const char* path, int id, size_t* sz)
{
	char name[256];
	snprintf(name, sizeof(name), "%s/output%d.raw", path, id);

	return nnt_load(name, sz);
}

static int ssd_compare(nn_t* nn, int id, float* output, size_t szo, float* gloden, size_t szg)
{
	int r = 0;
	int i;
	float IoU;
	int num_det = nn->network->outputs[0]->layer->C->context->nhwc.N;
	image_t* im;

	if(g_InputImagePath != NULL)
	{
		im = image_open(g_InputImagePath);
		assert(im != NULL);

		for(int i=0; i<num_det; i++)
		{
			float batch = output[7*i];
			int label = output[7*i+1];
			float prop = output[7*i+2];

			int x = output[7*i+3]*im->w;
			int y = output[7*i+4]*im->h;
			int w = output[7*i+5]*im->w-x;
			int h = output[7*i+6]*im->h-y;

			const char* name = "unknow";
			if(label < ARRAY_SIZE(voc_names_for_ssd))
			{
				name = voc_names_for_ssd[label];
			}

			printf("predict L=%s(%d) P=%.2f @%d %d %d %d\n", name, label, prop, x, y, w, h);
			image_draw_rectange(im, x, y, w, h, 0x00FF00);

			char text[128];

			snprintf(text, sizeof(text), "%s %.1f%%", name, prop*100);
			image_draw_text(im, x, y, text,  0xFF0000);
		}

		image_save(im, "predictions.png");
		printf("checking predictions.png for %s\n", g_InputImagePath);
#ifdef _WIN32
		system("predictions.png");
#else
		system("eog predictions.png");
#endif
		image_close(im);
	}
	else
	{
		EXPECT_EQ(num_det, szg/7);

		for(i=0; i<num_det; i++)
		{
			IoU = ssd::JaccardOverlap(&output[7*i+3], &gloden[7*i+3]);

			EXPECT_EQ(output[7*i], gloden[7*i]);	/* batch */
			EXPECT_EQ(output[7*i+1], gloden[7*i+1]); /* label */
			EXPECT_NEAR(output[7*i+2], gloden[7*i+2], 0.05); /* prop */
			EXPECT_GT(IoU, 0.9);

			if(output[7*i] != gloden[7*i])
			{
				r = -1;
			}

			if(output[7*i+1] != gloden[7*i+1])
			{
				r = -2;
			}

			if(std::fabs(output[7*i+2]-gloden[7*i+2]) > 0.05)
			{
				r = -3;
			}

			if(IoU < 0.9)
			{
				r = -4;
			}
		}

		if(0 != r)
		{
			printf("output for image %d is not correct\n", id);
		}
	}

	return r;
}

static int yolov3_compare(nn_t* nn, int id, float* output, size_t szo, float* gloden, size_t szg)
{
	int r = 0;
	image_t* im;
	layer_context_t* context = (layer_context_t*)nn->network->inputs[0]->layer->C->context;

	int netw = context->nhwc.W;
	int neth = context->nhwc.H;

	EXPECT_EQ(context->nhwc.C, 3);
	if(g_InputImagePath != NULL)
	{
		im = image_open(g_InputImagePath);
		assert(im != NULL);
		int num_det = nn->network->outputs[0]->layer->C->context->nhwc.N;

		for(int i=0; i<num_det; i++)
		{
			float batch = output[7*i];
			int label = output[7*i+1];
			float prop = output[7*i+2];

			int new_w=0;
			int new_h=0;
			if (((float)netw/im->w) < ((float)neth/im->h)) {
				new_w = netw;
				new_h = (im->h * netw)/im->w;
			} else {
				new_h = neth;
				new_w = (im->w * neth)/im->h;
			}

			float bx = output[7*i+3];
			float by = output[7*i+4];
			float bw = output[7*i+5];
			float bh = output[7*i+6];
			bx =  (bx - (netw - new_w)/2./netw) / ((float)new_w/netw);
			by =  (by - (neth - new_h)/2./neth) / ((float)new_h/neth);
			bw *= (float)netw/new_w;
			bh *= (float)neth/new_h;

			int x = bx*im->w;
			int y = by*im->h;
			int w = bw*im->w;
			int h = bh*im->h;

			x -= w/2;
			y -= h/2;

			const char* name = "unknow";
			if(label < ARRAY_SIZE(coco_names_for_yolo))
			{
				name = coco_names_for_yolo[label];
			}

			printf("predict L=%s(%d) P=%.2f @%d %d %d %d\n", name, label, prop, x, y, w, h);
			image_draw_rectange(im, x, y, w, h, 0x00FF00);

			char text[128];

			snprintf(text, sizeof(text), "%s %.1f%%", name, prop*100);
			image_draw_text(im, x, y, text, 0xFF0000);
		}

		image_save(im, "predictions.png");
		printf("checking predictions.png for %s\n", g_InputImagePath);
#ifdef _WIN32
		system("predictions.png");
#else
		system("eog predictions.png");
#endif
		image_close(im);
	}

	return r;
}

static int vehicle_attributes_recognition_barrier_0039_compare(nn_t* nn, int id, float* output, size_t szo, float* gloden, size_t szg)
{
	static const char* types[] = { "car", "bus", "truck", "van" };
	static const char* colors[] = { "white", "gray", "yellow", "red", "green", "blue", "black" };

	printf("type: ");
	for(int i=0; i<szo; i++)
	{
		printf("%s: %.3f, ", types[i], output[i]);
	}
	printf("\n");

	float* out = (float*)nn->network->outputs[1]->data;

	printf("color: ");
	for(int i=0; i<7; i++)
	{
		printf("%s: %.3f, ", colors[i], out[i]);
	}
	printf("\n");
	return 0;
}

static int enet_compare(nn_t* nn, int id, float* output, size_t szo, float* gloden, size_t szg)
{
	NHWC_t* nhwc = &(nn->network->outputs[0]->layer->C->context->nhwc);
	const char* labels = "gtest/models/enet/cityscapes19.png";
	image_t* color_im = image_open(labels);
	if(color_im != NULL) {
		image_t* im = image_create(nhwc->W, nhwc->H, 3);
		for(int h=0; h<nhwc->H; h++) {
			for(int w=0; w<nhwc->W; w++) {
				int index = 0;
				int offset = (h*nhwc->W+w)*nhwc->C;
				float max = output[offset];
				for(int c=1; c<nhwc->C; c++) {	/* argmax */
					float value = output[offset+c];
					if(value > max) {
						max = value;
						index = c;
					}
				}
				uint32_t color = (color_im->data[3*index]<<16) + (color_im->data[3*index+1]<<8) + color_im->data[3*index+2];
				image_draw_pixel(im, w, h, color);
			}
		}
		image_save(im, "predictions.png");
		printf("checking predictions.png for %s\n", g_InputImagePath?:"null");
		#ifdef _WIN32
		system("predictions.png");
		#else
		system("eog predictions.png");
		#endif
		image_close(im);
		image_close(color_im);
	} else {
		printf("file %s not exists\n", labels);
	}
	return 0;
}
static int kws_compare(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg)
{
	static const char* kws[] = {"Silence", "Unknown","yes","no","up","down","left","right","on","off","stop","go"};
	printf("kws: ");
	for(int i=0; i<szo; i++)
	{
		printf("%s: %.3f, ", kws[i], output[i]);
	}
	printf("\n");
	return 0;
}

static int ds_compare(nn_t* nn, int id, float * output, size_t szo, float* gloden, size_t szg)
{
	return 0;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
void ModelTestMain(runtime_type_t runtime,
		const network_t* network,
		const char* input,
		const char* output,
		const nnt_model_args_t* args,
		float mintop1)
{
	int r = 0;
	size_t x_test_sz;
	size_t y_test_sz;
	float* x_test = NULL;
	int32_t* y_test = NULL;

	const nn_input_t* const * inputs = network->inputs;
	const nn_output_t* const * outputs = network->outputs;

	int H,W,C,B;
	int classes;

	auto tcreate_s = std::chrono::high_resolution_clock::now();
	nn_t* nn = nn_create(network, runtime);
	ASSERT_TRUE(nn != NULL);
	auto tcreate_e = std::chrono::high_resolution_clock::now();
	double tcreate = std::chrono::duration_cast<std::chrono::nanoseconds>(tcreate_e-tcreate_s).count();

	double trun_sum = 0;
	size_t fps = 0;

	if(NULL == nn)
	{
		return;
	}
	/* This demo code is really a mess, didn't care about memory leak issue, just for test */
	H = inputs[0]->layer->C->context->nhwc.H;
	W = inputs[0]->layer->C->context->nhwc.W;
	C = inputs[0]->layer->C->context->nhwc.C;
	classes = NHWC_BATCH_SIZE(outputs[0]->layer->C->context->nhwc);
	if(NULL == args)
	{
		x_test = (float*)nnt_load(input, &x_test_sz);
		y_test = (int32_t*)nnt_load(output,&y_test_sz);
		B = x_test_sz/(H*W*C*sizeof(float));
		ASSERT_EQ(B, y_test_sz/sizeof(int32_t));
	}
	else if((args->flags&ARGS_PREFETCH_ALL) != 0)
	{
		x_test = (float*)args->load_input(nn, input, -1, &x_test_sz);
		B = x_test_sz/(H*W*C*sizeof(float));
	}
	else
	{
		B = args->n;
	}

	void* IN;

	size_t top1 = 0;
	for(int i=0; (i<B) && (r==0); i++)
	{
		if(g_CaseNumber != -1)
		{
			i = g_CaseNumber;
		}
		float* in;
		size_t sz_in;
		float* golden = NULL;
		size_t sz_golden;

		if((NULL == args) || ((args->flags&ARGS_PREFETCH_ALL) != 0))
		{
			in = x_test+H*W*C*i;
		}
		else
		{
			in = (float*)args->load_input(nn, input, i, &sz_in);
			if(inputs[0]->layer->dtype != L_DT_STRING) {
				EXPECT_EQ(sz_in, H*W*C*sizeof(float));
			}
			if(NULL == g_InputImagePath)
			{
				golden = (float*)args->load_output(input, i, &sz_golden);
				ASSERT_TRUE(golden != NULL);
			}
		}

		if(inputs[0]->layer->dtype == L_DT_STRING)
		{
			sz_in = sizeof(void*)*2;
			IN = in;
		}
		else if(network->type== NETWORK_TYPE_Q8)
		{
			sz_in = H*W*C;
			IN = nnt_quantize8(in, H*W*C, LAYER_Q(inputs[0]->layer));
			ASSERT_TRUE(IN != NULL);
		}
		else if(network->type== NETWORK_TYPE_S8)
		{
			sz_in = H*W*C;
			IN = nnt_quantize8(in, H*W*C, LAYER_Q(inputs[0]->layer),
						LAYER_Z(inputs[0]->layer),
						(float)LAYER_S(inputs[0]->layer)/NN_SCALER);
			ASSERT_TRUE(IN != NULL);
		}
		else if(network->type== NETWORK_TYPE_Q16)
		{
			sz_in = H*W*C*sizeof(int16_t);
			IN = nnt_quantize16(in, H*W*C, LAYER_Q(inputs[0]->layer));
			ASSERT_TRUE(IN != NULL);
		}
		else
		{
			sz_in = H*W*C*sizeof(float);
			IN = in;
		}

		memcpy(inputs[0]->data, IN, sz_in);

		auto trun_s = std::chrono::high_resolution_clock::now();
		r = nn_predict(nn);
		EXPECT_EQ(0, r);
		auto trun_e = std::chrono::high_resolution_clock::now();

		trun_sum = std::chrono::duration_cast<std::chrono::nanoseconds>(trun_e-trun_s).count();
		fps ++;

		if(0 == r)
		{
			int y=-1;
			float prob = 0;
			float* out = (float*)outputs[0]->data;
			if( (outputs[0]->layer->op == L_OP_DETECTIONOUTPUT) ||
				(outputs[0]->layer->op == L_OP_YOLOOUTPUT))
			{
				/* already in float format */
			}
			else if(network->type== NETWORK_TYPE_Q8)
			{
				out = nnt_dequantize8((int8_t*)out, classes, LAYER_Q(outputs[0]->layer));
			}
			else if(network->type== NETWORK_TYPE_S8)
			{
				out = nnt_dequantize8((int8_t*)out, classes, LAYER_Q(outputs[0]->layer),
						LAYER_Z(outputs[0]->layer),
						(float)LAYER_S(outputs[0]->layer)/NN_SCALER);
			}
			else if(network->type== NETWORK_TYPE_Q16)
			{
				out = nnt_dequantize16((int16_t*)out, classes, LAYER_Q(outputs[0]->layer));
			}

			if(NULL == args)
			{
				for(int j=0; j<classes; j++)
				{
					if(out[j] > prob)
					{
						prob = out[j];
						y = j;
					}
				}

				EXPECT_GE(y, 0);

				if(y == y_test[i])
				{
					top1 ++;
				}
			}
			else if((args->flags&ARGS_COMPARE_ALL_AT_END) != 0)
			{
				static float* final_o = NULL;
				if(0 == i) {
					final_o = (float*)malloc(B*classes*sizeof(float));
				}

				memcpy(&final_o[i*classes], out, classes*sizeof(float));

				if(i == (B-1)) {
					y = args->compare(nn, -1, final_o, B*classes, NULL, 0);
					free(final_o);
				}
			}
			else
			{
				y = args->compare(nn, i, out, classes, golden, sz_golden/sizeof(float));
				if(0 == y)
				{
					top1 ++;
				}

				free(in);
				if(NULL != golden) free(golden);
			}

			if(out != outputs[0]->data)
			{
				free(out);
			}

			if((g_CaseNumber != -1) || (g_InputImagePath != NULL))
			{
				if(NULL == args)
				{
					printf("image %d predict as %d%s%d with prob=%.2f\n", i, y, (y==y_test[i])?"==":"!=", y_test[i], prob);
				}
				break;
			}
		}

		if(IN != in)
		{
			free(IN);
		}

		if((i>0) && ((i%1000) == 0))
		{
			printf("LWNN TOP1 is %f on %d test images\n", (float)top1/i, i);
		}
	}

	if((-1 == g_CaseNumber) && (NULL == g_InputImagePath))
	{
		printf("LWNN TOP1 is %f\n", (float)top1/B);
		EXPECT_GT(top1, B*mintop1);
	}
	nn_destory(nn);

	if(NULL != x_test)
	{
		free(x_test);
	}
	if(NULL != y_test)
	{
		free(y_test);
	}

	printf("Create cost %.3fms , Inference cost avg %.3fms\n", tcreate/1000000, trun_sum/fps/1000000);
}

void NNTModelTestGeneral(runtime_type_t runtime,
		const char* netpath,
		const char* input,
		const char* output,
		const void* args,
		float mintop1,
		float not_found_okay)
{
	const network_t* network;
	void* dll;
	network = nnt_load_network(netpath, &dll);
	if(not_found_okay == FALSE)
	{
		EXPECT_TRUE(network != NULL);
	}
	if(network == NULL)
	{
		return;
	}
	ModelTestMain(runtime, network, input, output, (const nnt_model_args_t*)args, mintop1);
	dlclose(dll);
}

NNT_MODEL_TEST_ALL(MNIST)

NNT_MODEL_TEST_ALL(UCI_INCEPTION)

NNT_MODEL_TEST_ALL(SSD)

NNT_MODEL_TEST_ALL(YOLOV3)

NNT_MODEL_TEST_ALL(YOLOV3TINY)

NNT_MODEL_TEST_ALL(VEHICLE_ATTR)

NNT_MODEL_TEST_ALL(ENET)

NNT_MODEL_TEST_ALL(KWS)

NNT_MODEL_TEST_ALL(DS)
