/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "nn.h"
#ifndef DISABLE_RUNTIME_CPU_FLOAT
#include "../runtime_cpu.h"
#include "arm_math.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	int32_t magnitude_squared;
	int32_t window_size;
	int32_t stride;
	int32_t desired_samples;
	int32_t desired_channels;
	int32_t upper_frequency_limit;
	int32_t lower_frequency_limit;
	int32_t dct_coefficient_count;
	int32_t filterbank_channel_count;
} mfcc_param_t;

typedef struct {
	LAYER_CPU_CONTEXT_MEMBER;
	const mfcc_param_t* param;
	float* frame;
	float* buffer;
	float* mel_energies;
	float* window_func;
	int32_t* fbank_filter_first;
	int32_t* fbank_filter_last;
	float** mel_fbank;
	float* dct_matrix;
	int frame_len_padded;
	arm_rfft_fast_instance_f32 rfft;
} layer_cpu_float_mfcc_context_t;

typedef struct
{
	void* data;
	size_t size;
} wav_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static void mfcc_compute(
		layer_cpu_float_mfcc_context_t* context,
		const int16_t* audio_data,
		float* mfcc_out)
{
	int32_t i, j, bin;
	int frame_len = context->param->window_size;
	int num_mfcc_features = context->nhwc.C;

	/* TensorFlow way of normalizing .wav data to (-1,1) */
	for (i = 0; i < frame_len; i++) {
		context->frame[i] = (float) audio_data[i] / (1 << 15);
	}
	/* Fill up remaining with zeros */
	memset(&context->frame[frame_len], 0,
			sizeof(float) * (context->frame_len_padded - frame_len));

	for (i = 0; i < frame_len; i++) {
		context->frame[i] *= context->window_func[i];
	}

	/* Compute FFT */
	arm_rfft_fast_f32(&context->rfft, context->frame, context->buffer, 0);

	/* Convert to power spectrum */
	/* frame is stored as [real0, realN/2-1, real1, im1, real2, im2, ...] */
	int32_t half_dim = context->frame_len_padded / 2;
	float first_energy = context->buffer[0] * context->buffer[0], last_energy = context->buffer[1]
			* context->buffer[1];  /* handle this special case */
	for (i = 1; i < half_dim; i++) {
		float real = context->buffer[i * 2], im = context->buffer[i * 2 + 1];
		context->buffer[i] = real * real + im * im;
	}
	context->buffer[0] = first_energy;
	context->buffer[half_dim] = last_energy;

	float sqrt_data;
	/* Apply mel filterbanks */
	for (bin = 0; bin < context->param->filterbank_channel_count; bin++) {
		j = 0;
		float mel_energy = 0;
		int32_t first_index = context->fbank_filter_first[bin];
		int32_t last_index = context->fbank_filter_last[bin];
		for (i = first_index; i <= last_index; i++) {
			arm_sqrt_f32(context->buffer[i], &sqrt_data);
			mel_energy += (sqrt_data) * context->mel_fbank[bin][j++];
		}
		context->mel_energies[bin] = mel_energy;

		/* avoid log of zero */
		if (mel_energy == 0.0)
			context->mel_energies[bin] = FLT_MIN;
	}

	/* Take log */
	for (bin = 0; bin < context->param->filterbank_channel_count; bin++)
		context->mel_energies[bin] = logf(context->mel_energies[bin]);

	//Take DCT. Uses matrix mul.
	for (i = 0; i < num_mfcc_features; i++) {
		float sum = 0.0;
		for (j = 0; j < context->param->filterbank_channel_count; j++) {
			sum += context->dct_matrix[i * context->param->filterbank_channel_count + j] * context->mel_energies[j];
		}
		mfcc_out[i] = sum;
	}
}
static int extract_features(layer_cpu_float_mfcc_context_t* context,
						wav_t* wav)
{
	int num_frames = context->nhwc.H;
	int num_mfcc_features = context->nhwc.C;
	int frame_shift = context->param->stride;
	int i;

	int16_t* wav_data = (int16_t*)wav->data;
	float* mfcc_out = (float*)context->out[0];

	for(i=0; i<num_frames; i++)
	{
		mfcc_compute(context, &wav_data[i*frame_shift], &mfcc_out[i*num_mfcc_features]);
	}

	return 0;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_MFCC_init(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_mfcc_context_t* context;
	int frame_len;
	int r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_mfcc_context_t), sizeof(float));

	if(0 == r)
	{
		context = (layer_cpu_float_mfcc_context_t*)layer->C->context;
		context->param = (const mfcc_param_t*)layer->blobs[0]->blob;
		frame_len = context->param->window_size;
		context->frame_len_padded = pow(2,ceil((log(frame_len)/log(2))));
		arm_rfft_fast_init_f32(&context->rfft, context->frame_len_padded);
	}

	return r;
}
int layer_cpu_float_MFCC_execute(const nn_t* nn, const layer_t* layer)
{
	int r = 0;
	layer_cpu_float_mfcc_context_t* context = (layer_cpu_float_mfcc_context_t*)layer->C->context;
	layer_cpu_context_t* input_context = (layer_cpu_context_t*)layer->inputs[0]->C->context;

	wav_t* wav = input_context->out[0];

	NNLOG(NN_DEBUG, ("execute %s: wav_data %d@%p\n",layer->name, (int)wav->size, wav->data));

	r = extract_features(context, wav);

	return r;
}
void layer_cpu_float_MFCC_deinit(const nn_t* nn, const layer_t* layer)
{
	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
