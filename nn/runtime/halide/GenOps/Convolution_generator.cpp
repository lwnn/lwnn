/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020 Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "GenCommon.h"
/* ============================ [ MACROS    ] ====================================================== */
/* ============================ [ TYPES     ] ====================================================== */
/* ref https://github.com/halide/Halide/blob/master/apps/nn_ops/Convolution_generator.cpp
 * ref https://github.com/halide/Halide/blob/master/apps/conv_layer/conv_layer_generator.cpp */
class ConvolutionLayer HL_FARTHER(ConvolutionLayer) {
public:
	HL_INPUT_BUFFER(input, 4);
	HL_INPUT_BUFFER(W, 4);
	HL_INPUT_BUFFER(B, 1);
	HL_OUTPUT_BUFFER(conv, 4);
	HL_INPUT_INT(strideX);
	HL_INPUT_INT(strideY);
	HL_INPUT_INT(padX);
	HL_INPUT_INT(padY);
	HL_INPUT_INT(iC);
	#ifdef LWNN_ON_HALIDE
	int iH, iW, knlX, knlY;
	#endif

	void generate() {
		#ifndef LWNN_ON_HALIDE
		Expr iH = input.dim(2).extent();
		Expr iW = input.dim(1).extent();

		Expr knlX = W.dim(1).extent();
		Expr knlY = W.dim(2).extent();
		#endif
		Var x("x"), y("y"), c("c"), n("n");

		Halide::RDom r(0, knlX, 0, knlY, 0, iC);

		Halide::Func in_bounded =
			Halide::BoundaryConditions::constant_exterior(input, 0.f,
					{{Halide::Expr(), Halide::Expr()},
					 {0, iW},
					 {0, iH},
					 {Halide::Expr(), Halide::Expr()}});
		conv(c, x, y, n) = B(c);
		Halide::Expr in_row = strideY * y + r.y - padY;
		Halide::Expr in_col = strideX * x + r.x - padX;
		conv(c, x, y, n) += W(r.z, r.x, r.y, c) * in_bounded(r.z, in_col, in_row, n);
	}
};
/* ============================ [ DECLARES  ] ====================================================== */
HL_REGISTER_GENERATOR(Convolution)
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
