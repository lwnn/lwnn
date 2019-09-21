/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2019  Parai Wang <parai@foxmail.com>
 */
#ifndef NN_THIRD_PARTY_IMAGE_IMAGE_H_
#define NN_THIRD_PARTY_IMAGE_IMAGE_H_
/* ============================ [ INCLUDES  ] ====================================================== */
#include <stdint.h>
/* ============================ [ MACROS    ] ====================================================== */
#ifdef __cplusplus
extern "C" {
#endif
/* ============================ [ TYPES     ] ====================================================== */
typedef struct {
	int w;
	int h;
	int c;
	uint8_t *data;
} image_t;
/* ============================ [ DECLARES  ] ====================================================== */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
image_t* image_open(const char *filename);
image_t* image_create(int w, int h, int c);
void image_close(image_t *im);
void image_save(image_t* im, const char *filename);
void image_fill(image_t* im, uint8_t color);
image_t* image_resize(image_t* im, int w, int h);
image_t* image_letterbox(image_t* im, int w, int h);

void image_draw(image_t* dest, image_t* source, int dx, int dy);
void image_draw_pixel(image_t* im, int x, int y, uint32_t color);
void image_fill_area(image_t* im, int x, int y, int cx, int cy, uint32_t color);
void image_draw_line(image_t* im, int x0, int y0, int x1, int y1, uint32_t color);
void image_draw_rectange(image_t* im, int x, int y, int w, int h, uint32_t color);
void image_draw_text(image_t* im, int x, int y, const char *string, uint32_t color);
void image_draw_char(image_t* im, int x, int y, uint8_t c, uint32_t color);
#ifdef __cplusplus
}
#endif
#endif /* NN_THIRD_PARTY_IMAGE_IMAGE_H_ */
