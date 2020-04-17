/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "runtime_halide.h"
#include "runtime_cpu.h"
/* ============================ [ MACROS    ] ====================================================== */
#ifdef _WIN32
#define BUILD_DIR "build/nt/"
#define DLLFIX ".dll"
#define LIBFIX ""
#else
#define BUILD_DIR "build/posix/"
#define DLLFIX ".so"
#define LIBFIX "lib"
#endif

#define FALLBACK_LAYER_OPS_HALIDE(op, to) FALLBACK_LAYER_OPS(halide, op, to)
/* ============================ [ TYPES     ] ====================================================== */
typedef struct
{
	int dummy;
} rte_halide_t;
/* ============================ [ DECLARES  ] ====================================================== */
extern "C" {
#define OP_DEF(op) L_OPS_DECLARE(halide_##op);
#include "opdef.h"
#undef OP_DEF

static inline void layer_halide_to_cpu_float_init_common(const nn_t*, const layer_t*) {}
static inline int layer_halide_to_cpu_float_pre_execute_common(const nn_t*, const layer_t*) { return 0; }
static inline void layer_halide_to_cpu_float_post_execute_common(const nn_t*, const layer_t*) {}
static inline void layer_halide_to_cpu_float_deinit_common(const nn_t*, const layer_t*) {}

FALLBACK_LAYER_OPS_HALIDE(INPUT, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(OUTPUT, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(MAXIMUM, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(RELU, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(MAXPOOL, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(RESHAPE, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(DENSE, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(SOFTMAX, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(PAD, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(DWCONV2D, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(CONCAT, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(AVGPOOL, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(ADD, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(CONST, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(DETECTIONOUTPUT, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(UPSAMPLE, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(YOLO, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(YOLOOUTPUT, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(DECONV2D, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(BATCHNORM, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(DILCONV2D, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(PRELU, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(MFCC, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(LSTM, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(MINIMUM, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(TRANSPOSE, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(DETECTION, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(PROPOSAL, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(PYRAMID_ROI_ALIGN, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(SLICE, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(MUL, cpu_float)
FALLBACK_LAYER_OPS_HALIDE(CLIP, cpu_float)
} /* extern "C"  end */
/* ============================ [ DATAS     ] ====================================================== */
static const layer_ops_t halide_lops[] =
{
#define OP_DEF(op) L_OPS_REF(halide_##op),
	#include "opdef.h"
#undef OP_DEF
};
/* ============================ [ LOCALS    ] ====================================================== */
static int halide_init_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;

	if(layer->op < ARRAY_SIZE(halide_lops))
	{
		r = halide_lops[layer->op].init(nn, layer);
	}

	return r;
}
static int halide_execute_layer(const nn_t* nn, const layer_t* layer)
{
	int r = NN_E_INVALID_LAYER;

	if(layer->op < ARRAY_SIZE(halide_lops))
	{
		NNLOG(NN_DEBUG, ("execute %s: [%d %d %d %d]\n", layer->name, L_SHAPES(layer)));
		r = halide_lops[layer->op].execute(nn, layer);
		NNDDO(NN_DEBUG, rte_ddo_save(nn, layer));
	}

	return r;
}

static int halide_deinit_layer(const nn_t* nn, const layer_t* layer)
{
	if(layer->op < ARRAY_SIZE(halide_lops))
	{
		halide_lops[layer->op].deinit(nn, layer);
	}

	return 0;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
extern "C" {
runtime_t rte_HALIDE_create(const nn_t* nn)
{
	rte_halide_t* rt = NULL;

	rt = (rte_halide_t*)malloc(sizeof(rte_halide_t));

	return rt;
}

int rte_HALIDE_init(const nn_t* nn)
{
	int r;
	rte_halide_t* rt = (rte_halide_t*)nn->runtime;

	r = rte_do_for_each_layer(nn, halide_init_layer);

	return r;
}

int rte_HALIDE_execute(const nn_t* nn)
{
	int r;
	rte_halide_t* rt = (rte_halide_t*)nn->runtime;

	r = rte_do_for_each_layer(nn, halide_execute_layer);

	return r;
}

void rte_HALIDE_destory(const nn_t* nn)
{
	rte_halide_t* rt = (rte_halide_t*)nn->runtime;

	rte_do_for_each_layer(nn, halide_deinit_layer);
}
}; /* extern "C"  end */


int rte_halide_create_layer_context(
			const nn_t* nn, const layer_t* layer,
			size_t sz, size_t nout)
{
	return rte_cpu_create_layer_context(nn, layer, sz, nout);
}

void rte_halide_destory_layer_context(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

int rte_halide_create_layer_common(const nn_t* nn, const layer_t* layer, size_t ctx_sz)
{
	return rte_cpu_create_layer_common(nn, layer, ctx_sz, sizeof(float));
}

Halide::Buffer<float>* rte_halide_create_buffer(const int* dims, const float* data) {
	Halide::Buffer<float>* buf = NULL;
	int dim = 0;
	size_t size = 1;

	while((dims[dim] != 0) && (dim < 4)) {
		size = size*dims[dim];
		dim ++;
	};

	switch(dim) {
		case 1:
			buf = new Halide::Buffer<float>(dims[0]);
			break;
		case 2:
			buf = new Halide::Buffer<float>(dims[1], dims[0]);
			break;
		case 3:
			buf = new Halide::Buffer<float>(dims[2], dims[1], dims[0]);
			break;
		case 4:
			buf = new Halide::Buffer<float>(dims[3], dims[2], dims[1], dims[0]);
			break;
		default:
			break;
	}

	if(NULL != buf) {
		buf->allocate();
		if(data != NULL) {
			std::copy(data, data+size, buf->begin());
		}
	}

	return buf;
}

void* halide_load_algorithm(const char* algo, void** dll)
{
	void* sym = NULL;
	std::string path = std::string(BUILD_DIR "nn/runtime/halide/ops/" LIBFIX) + algo + DLLFIX;

	*dll = dlopen(path.c_str(), RTLD_NOW);
	if((*dll) != NULL) {
		sym = dlsym(*dll, algo);
		if(NULL == sym) {
			dlclose(*dll);
			*dll = NULL;
		}
	}

	NNLOG(NN_DEBUG, ("load %s %s\n", path.c_str(), (sym!=NULL)?"okay":"fail, will use JIT"));

	return sym;
}
