/**
 * LWNN - Lightweight Neural Network
 * Copyright (C) 2020 Parai Wang <parai@foxmail.com>
 */
/* ============================ [ INCLUDES  ] ====================================================== */
#include "Halide.h"
/* ============================ [ MACROS    ] ====================================================== */
using namespace Halide;
/* ============================ [ TYPES     ] ====================================================== */
/* ref https://github.com/halide/Halide/blob/master/apps/nn_ops/Convolution_generator.cpp
 * ref https://github.com/halide/Halide/blob/master/apps/conv_layer/conv_layer_generator.cpp */
class ConvolutionLayer : public Halide::Generator<ConvolutionLayer> {
public:
	Input<Buffer<float>> input{"input", 4};
	Input<Buffer<float>> W{"weights", 4};
	Input<Buffer<float>> B{"bias", 1};
	Output<Buffer<float>> conv{"conv2d", 4};
	Input<int> strideX{"strideX"};
	Input<int> strideY{"strideY"};
	Input<int> padX{"padX"};
	Input<int> padY{"padY"};
	Input<int> iC{"input_depth"};

	void generate() {
		Expr iH = input.dim(1).extent();
		Expr iW = input.dim(2).extent();

		Expr knlX = W.dim(2).extent();
		Expr knlY = W.dim(1).extent();

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
HALIDE_REGISTER_GENERATOR(ConvolutionLayer, conv2d)
/*/path/to/hlGenOps -g conv2d -o nn/runtime/halide/ops/ target=x86-64-linux-avx-avx2-f16c-fma-sse41 */
/* ============================ [ DATAS     ] ====================================================== */
/* ============================ [ LOCALS    ] ====================================================== */
/* ============================ [ FUNCTIONS ] ====================================================== */
