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
extern const uint8_t pCharset10x14[];
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
		} else {
			im->c = 3;
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

void image_draw(image_t* dest, image_t* source, int dx, int dy)
{
	int y;

	assert((dx>=0) && (dx<dest->w));
	assert((dy>=0) && (dy<dest->h));

	assert(((dx+source->w)<=dest->w));
	assert(((dy+source->h)<=dest->h));

	for(y=0; y<source->h; y++) {
		memcpy(&dest->data[((dy+y)*dest->w+dx)*dest->c],
				&source->data[y*source->w*source->c], source->w*source->c);
	}
}

void image_draw_pixel(image_t* im, int x, int y, uint32_t color)
{
	uint32_t alpha = 0xFF&(color>>24);
	if( (x>=0) && (y>=0) &&
		(x<im->w) && (y<im->h) ) {
		if(3 == im->c) {
			if(0 == alpha) {
				im->data[(y*im->w+x)*3] = 0xFF&(color>>16);
				im->data[(y*im->w+x)*3+1] = 0xFF&(color>>8);
				im->data[(y*im->w+x)*3+2] = 0xFF&(color);
			} else {
				im->data[(y*im->w+x)*3] = ((0xFF&(color>>16))*alpha + im->data[(y*im->w+x)*3]*(255-alpha))/255;
				im->data[(y*im->w+x)*3+1] = ((0xFF&(color>>8))*alpha + im->data[(y*im->w+x)*3+1]*(255-alpha))/255;
				im->data[(y*im->w+x)*3+2] = ((0xFF&(color))*alpha + im->data[(y*im->w+x)*3+2]*(255-alpha))/255;
			}
		} else if(1 == im->c) {
			im->data[(y*im->w+x)*3] = 0xFF&(color);
		}
	}
}

void image_fill_area(image_t* im, int x, int y, int cx, int cy, uint32_t color)
{
	int x0, x1, y1;

	x0 = x;
	x1 = x + cx;
	y1 = y + cy;
	for(; y < y1; y++) {
		for(x = x0; x < x1; x++) {
			image_draw_pixel(im, x, y, color);
		}
	}
}

void image_draw_line(image_t* im, int x0, int y0, int x1, int y1, uint32_t color)
{
	int dy, dx;
	int addx, addy;
	int P, diff, i;


	/* speed improvement if vertical or horizontal */
	if (x0 == x1) {
		if (y1 > y0) {
			image_fill_area(im, x0, y0, 1, y1-y0+1, color);
		}
		else {
			image_fill_area(im, x0, y1, 1, y0-y1+1, color);
		}
		return;
	}
	if (y0 == y1) {
		if (x1 > x0)
			image_fill_area(im, x0, y0, x1-x0+1, 1, color);
		else
			image_fill_area(im, x1, y0, x0-x1+1, 1, color);
		return;
	}

	if (x1 >= x0) {
		dx = x1 - x0;
		addx = 1;
	} else {
		dx = x0 - x1;
		addx = -1;
	}
	if (y1 >= y0) {
		dy = y1 - y0;
		addy = 1;
	} else {
		dy = y0 - y1;
		addy = -1;
	}

	if (dx >= dy) {
		dy *= 2;
		P = dy - dx;
		diff = P - dx;

		for(i=0; i<=dx; ++i) {
			image_draw_pixel(im, x0, y0, color);
			if (P < 0) {
				P  += dy;
				x0 += addx;
			} else {
				P  += diff;
				x0 += addx;
				y0 += addy;
			}
		}
	} else {
		dx *= 2;
		P = dx - dy;
		diff = P - dy;

		for(i=0; i<=dy; ++i) {
			image_draw_pixel(im, x0, y0, color);
			if (P < 0) {
				P  += dx;
				y0 += addy;
			} else {
				P  += diff;
				x0 += addx;
				y0 += addy;
			}
		}
	}
}

void image_draw_rectange(image_t* im, int x, int y, int w, int h, uint32_t color)
{
	image_draw_line(im, x, y, x+w, y, color);
	image_draw_line(im, x, y, x, y+h, color);
	image_draw_line(im, x, y+h, x+w, y+h, color);
	image_draw_line(im, x+w, y, x+w, y+h, color);
}

image_t* image_letterbox(image_t* im, int w, int h, uint8_t fill_color)
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
	image_fill(boxed, fill_color);
	image_draw(boxed, resized, (w-new_w)/2, (h-new_h)/2);
	image_close(resized);

	return boxed;
}


void image_draw_char(image_t* im, int x, int y, uint8_t c, uint32_t color)
{
	int row, col ;

	assert( (c >= 0x20) && (c <= 0x7F) ) ;

	for (col = 0; col < 10; col++)
	{
		for (row = 0 ; row < 8 ; row++)
		{
			if ( (pCharset10x14[((c - 0x20) * 20) + col * 2] >> (7 - row)) & 0x1 )
			{
				image_draw_pixel(im, x+col, y+row, color);
			}
		}

		for ( row = 0 ; row < 6 ; row++ )
		{
			if ( (pCharset10x14[((c - 0x20) * 20) + col * 2 + 1] >> (7 - row)) & 0x1 )
			{
				image_draw_pixel(im, x+col, y+row+8, color);
			}
		}
	}
}

void image_draw_text(image_t* im, int x, int y, const char *string, uint32_t color)
{
	unsigned xorg = x;

	while ( *string != 0 )
	{
		if ( *string == '\n' )
		{
			y += 14 + 2 ;
			x = xorg ;
		}
		else
		{
			image_draw_char(im, x, y, *string, color) ;
			x += 10 + 2;
		}

		string++;
	}
}
