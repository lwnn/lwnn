/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include <stdio.h>
#include <string.h>
#include <assert.h>
#include "image.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb/stb_image.h"
#define STB_IMAGE_RESIZE_IMPLEMENTATION
#include "stb/stb_image_resize.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb/stb_image_write.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
image_t* image_open(const char *filename)
{
	image_t* im = malloc(sizeof(image_t));

	if(NULL != im) {
		im->data = (void*)stbi_load(filename, &im->w, &im->h, &im->c, 3);
		if (NULL == im->data) {
			printf("Cannot load image \"%s\"\nSTB Reason: %s\n", filename, stbi_failure_reason());
			free(im);
			im = NULL;
		}
	}

	return im;
}

image_t* image_create(int w, int h, int c)
{
	image_t* im = malloc(sizeof(image_t));

	if(NULL != im) {
		im->data = malloc(w*h*c);
		if (NULL == im->data) {
			free(im);
			im = NULL;
		} else {
			im->w = w;
			im->h = h;
			im->c = c;
		}
	}

	return im;
}

void image_close(image_t *im)
{
	free(im->data);
	free(im);
}

void image_save(image_t* im, const char *filename)
{
	int success = 0;
	success = stbi_write_png(filename, im->w, im->h, im->c, im->data, im->w*im->c);

	if(!success) {
		fprintf(stderr, "Failed to write image %s\n", filename);
	}
}


void image_fill(image_t* im, uint8_t color)
{
	memset(im->data, color, im->w*im->h*im->c);
}

image_t* image_resize(image_t* im, int w, int h)
{
	int r;
	image_t* resized = image_create(w, h, im->c);

	if(NULL != resized)
	{
		r = stbir_resize_uint8(im->data, im->w, im->h, 0,
				resized->data, resized->w, resized->h, 0, im->c);

		if(1 != r)
		{
			image_close(resized);
			resized = NULL;
		}
	}

	return resized;
}

void image_embed(image_t* dest, image_t* source)
{
	int y;

	int dx = (dest->w-source->w)/2;
	int dy = (dest->h-source->h)/2;

	for(y=0; y<source->h; y++){
		memcpy(&dest->data[((dy+y)*dest->w+dx)*dest->c],
				&source->data[y*source->w*source->c], source->w*source->c);
	}
}

image_t* image_letterbox(image_t* im, int w, int h)
{
	image_t* boxed;
	image_t* resized;
	int new_w = im->w;
	int new_h = im->h;

	if ((w*1000/im->w) < (h*1000/im->h)) {
		new_w = w;
		new_h = (im->h * w)/im->w;
	} else {
		new_h = h;
		new_w = (im->w * h)/im->h;
	}

	resized = image_resize(im, new_w, new_h);
	boxed = image_create(w, h, im->c);
	image_fill(boxed, 127);
	image_embed(boxed, resized);
	image_close(resized);

	return boxed;
}
