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
#define M_2PI 6.283185307179586476925286766559005
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
	int max_frames;
} layer_cpu_float_mfcc_context_t;

typedef struct
{
	void* data;
	size_t size;
} wav_t;
/* ============================ [ DECLARES  ] ====================================================== */
extern void layer_cpu_float_MFCC_deinit(const nn_t* nn, const layer_t* layer);
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
static inline float MelScale(float freq) {
	return 1127.0f * logf (1.0f + freq / 700.0f);
}

float* create_dct_matrix(int32_t input_length, int32_t coefficient_count) {
	int32_t k, n;
	float * M = malloc(sizeof(float)*input_length*coefficient_count);
	float normalizer;
	if(NULL != M) {
		arm_sqrt_f32(2.0/(float)input_length,&normalizer);
		for (k = 0; k < coefficient_count; k++) {
			for (n = 0; n < input_length; n++) {
				M[k*input_length+n] = normalizer * cos( ((double)M_PI)/input_length * (n + 0.5) * k );
			}
		}
	}
	return M;
}

int create_mel_fbank(layer_cpu_float_mfcc_context_t* context) {

	int r = 0;
	int32_t bin, i;

	int32_t num_fft_bins = context->frame_len_padded/2;
	float fft_bin_width = ((float)context->param->desired_samples) / context->frame_len_padded;
	float mel_low_freq = MelScale(context->param->lower_frequency_limit);
	float mel_high_freq = MelScale(context->param->upper_frequency_limit);
	float mel_freq_delta = (mel_high_freq - mel_low_freq) / (context->param->filterbank_channel_count+1);

	float *this_bin = malloc(sizeof(float)*num_fft_bins);

	if(NULL == this_bin) {
		r = NN_E_NO_MEMORY;
	}

	context->mel_fbank = malloc(sizeof(float*)*context->param->filterbank_channel_count);

	if(NULL == context->mel_fbank) {
		free(this_bin);
		r = NN_E_NO_MEMORY;
	} else {
		memset(context->mel_fbank, 0, sizeof(float*)*context->param->filterbank_channel_count);
	}

	for (bin = 0; (0 == r) && (bin < context->param->filterbank_channel_count); bin++) {

		float left_mel = mel_low_freq + bin * mel_freq_delta;
		float center_mel = mel_low_freq + (bin + 1) * mel_freq_delta;
		float right_mel = mel_low_freq + (bin + 2) * mel_freq_delta;

		int32_t first_index = -1, last_index = -1;

		for (i = 0; i < num_fft_bins; i++) {

			float freq = (fft_bin_width * i);  /* center freq of this fft bin. */
			float mel = MelScale(freq);
			this_bin[i] = 0.0;

			if (mel > left_mel && mel < right_mel) {
				float weight;
				if (mel <= center_mel) {
					weight = (mel - left_mel) / (center_mel - left_mel);
				} else {
					weight = (right_mel-mel) / (right_mel-center_mel);
				}
				this_bin[i] = weight;
				if (first_index == -1)
					first_index = i;
					last_index = i;
			}
		}

		context->fbank_filter_first[bin] = first_index;
		context->fbank_filter_last[bin] = last_index;
		context->mel_fbank[bin] = malloc(sizeof(float)*(last_index-first_index+1));
		if(NULL == context->mel_fbank[bin]) {
			r = NN_E_NO_MEMORY;
		}

		int32_t j = 0;
		/* copy the part we care about */
		for (i = first_index; (0 == r) && (i <= last_index); i++) {
			context->mel_fbank[bin][j++] = this_bin[i];
		}
	}

	if(0 == r) {
		free(this_bin);
	}
	return r;
}

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

	/* Take DCT. Uses matrix mul. */
	for (i = 0; i < num_mfcc_features; i++) {
		float sum = 0.0;
		for (j = 0; j < context->param->filterbank_channel_count; j++) {
			sum += context->dct_matrix[i * context->param->filterbank_channel_count + j] * context->mel_energies[j];
		}
		mfcc_out[i] = sum;
	}
}
static int extract_features(const layer_t* layer, layer_cpu_float_mfcc_context_t* context,
						wav_t* wav)
{
	int r = 0;
	int num_frames = context->nhwc.H;
	int nframes;
	int num_mfcc_features = context->nhwc.C;
	int frame_shift = context->param->stride;
	int i;

	int16_t* wav_data = (int16_t*)wav->data;
	float* mfcc_out = (float*)context->out[0];

	nframes = (wav->size/2)/frame_shift;

	if(-1 == layer->dims[1]) { /* dynamic number of features */
		if(NULL == mfcc_out) {
			mfcc_out = malloc(sizeof(float)*nframes*num_mfcc_features);
			context->max_frames = nframes;
		} else if(nframes > context->max_frames) {
			free(mfcc_out);
			mfcc_out = malloc(sizeof(float)*nframes*num_mfcc_features);
			context->max_frames = nframes;
		} else {
			/* old memory is enough */
		}

		if(NULL == mfcc_out) {
			r = NN_E_NO_MEMORY;
		} else {
			num_frames = nframes;
			context->nhwc.H = num_frames;
		}
		context->out[0] = mfcc_out;
	} else {
		if(nframes < num_frames) {
			r = NN_E_INPUT_TOO_SMALL;
		}
	}

	if(0 == r) {
		for(i=0; i<num_frames; i++) {
			mfcc_compute(context, &wav_data[i*frame_shift], &mfcc_out[i*num_mfcc_features]);
		}
	}

	return r;
}
/* ============================ [ FUNCTIONS ] ====================================================== */
int layer_cpu_float_MFCC_init(const nn_t* nn, const layer_t* layer)
{
	layer_cpu_float_mfcc_context_t* context;
	size_t scratch_size = 0;
	int frame_len;
	int r = rte_cpu_create_layer_common(nn, layer, sizeof(layer_cpu_float_mfcc_context_t), sizeof(float));

	if(0 == r) {
		context = (layer_cpu_float_mfcc_context_t*)layer->C->context;
		memset(&((layer_cpu_context_t*)context)[1], 0,
				sizeof(layer_cpu_float_mfcc_context_t)-sizeof(layer_cpu_context_t));

		context->param = (const mfcc_param_t*)layer->blobs[0]->blob;
		frame_len = context->param->window_size;
		context->frame_len_padded = pow(2,ceil((log(frame_len)/log(2))));
		scratch_size += sizeof(float)*context->frame_len_padded;
		scratch_size += sizeof(float)*context->frame_len_padded;
		scratch_size += sizeof(float)*context->param->filterbank_channel_count;
		nn_request_scratch(nn, scratch_size);
		/* create window function */
		context->window_func = (float*)malloc(sizeof(float)*frame_len);
		if(NULL != context->window_func) {
			for (int i = 0; i < frame_len; i++)
				context->window_func[i] = 0.5 - 0.5*cos(M_2PI * ((float)i) / (frame_len));
		} else {
			r = NN_E_NO_MEMORY;
		}

		arm_rfft_fast_init_f32(&context->rfft, context->frame_len_padded);
	}

	if(0 == r) {
		/* create mel filterbank */
		context->fbank_filter_first = (int32_t*)malloc(sizeof(int32_t)*context->param->filterbank_channel_count);
		context->fbank_filter_last = (int32_t*)malloc(sizeof(int32_t)*context->param->filterbank_channel_count);
		if( (NULL != context->fbank_filter_first) &&
			(NULL != context->fbank_filter_last)) {
			r = create_mel_fbank(context);
		}
	}

	if(0 == r) {
		/* create DCT matrix */
		context->dct_matrix = create_dct_matrix(context->param->filterbank_channel_count, context->nhwc.C);
		if(NULL == context->dct_matrix) {
			r = NN_E_NO_MEMORY;
		}
	}

	if(0 != r)
	{
		layer_cpu_float_MFCC_deinit(nn, layer);
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

	context->frame = (float*)nn->scratch.area;
	context->buffer = context->frame + context->frame_len_padded;
	context->mel_energies = context->buffer + context->frame_len_padded;

	r = extract_features(layer, context, wav);

	return r;
}
void layer_cpu_float_MFCC_deinit(const nn_t* nn, const layer_t* layer)
{
	int i;
	layer_cpu_float_mfcc_context_t* context = (layer_cpu_float_mfcc_context_t*)layer->C->context;

	if(NULL != context) {
		if((-1 == layer->dims[1]) && (NULL != context->out[0])) {
			free(context->out[0]);
		}
		if(NULL != context->dct_matrix) free(context->dct_matrix);
		if(NULL != context->mel_fbank) {
			for (i = 0; i < context->param->filterbank_channel_count; i++) {
				if(NULL != context->mel_fbank[i]) free(context->mel_fbank[i]);
			}
			free(context->mel_fbank);
		}

		if(NULL != context->window_func) free(context->window_func);
		if(NULL != context->fbank_filter_first) free(context->fbank_filter_first);
		if(NULL != context->fbank_filter_last) free(context->fbank_filter_last);
	}

	rte_cpu_destory_layer_context(nn, layer);
}

#endif /* DISABLE_RUNTIME_CPU_FLOAT */
