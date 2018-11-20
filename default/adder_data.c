/*
 * adder_data.c
 *
 *  Created on: Feb 26, 2018
 *      Author: keyan
 */

#include "adder.h"
#include "ann_weights.h"
// #include "libfixmath/fix16.c"
// #include "libfixmath/fix16_exp.c"

// ANN_F newadder;
// ANN_F * adderANN = &newadder;
ANN newadder;
ANN * adderANN = &newadder;

// fix16_t hidden[NEURONS2];

void init()
{
	int i, j, num_weights;
	int layers[3] = {2, 2, 1};
	int activation[2] = {1, 1};
	ann_init_file(adderANN, 3, layers, 1, activation, "adder.net");
	// INIT_ANN_CUSTOM_FIXED(adderANN, 3, layers, weights, 1, activation, i, j, num_weights, MAX_WEIGHTS_BIAS, MAX_LAYERS, MAX_LAYERS_LESS1)
}

float add(int x, int y)
{
	printf("Inputs: %d + %d\n", x, y);
	// fix16_t output[1];
	// fix16_t inputs[2] = {(fix16_t)(x * 100), (fix16_t)(y * 100)};

	// int i, j, k;
	// fix16_t weighted_sum;

	// RUN_ANN_LAYER_FIXED(inputs, hidden, adderANN, 1, i, j, k, weighted_sum, NEURONS1, NEURONS2)
	// RUN_ANN_LAYER_FIXED(hidden, output, adderANN, 2, i, j, k, weighted_sum, NEURONS2, NEURONS3)

	// printf("In: [%d	%d]\n", inputs[0]/100, inputs[1]/100);
	// printf("Out: %d\n", output[0]/100);

	float quotient = 2 * RANGE;
	float inputs[2] = {((float)x)/quotient, ((float)y)/quotient};
	float outputs[1];
	ann_run(inputs, outputs, adderANN);

	return outputs[0]*quotient;
}

