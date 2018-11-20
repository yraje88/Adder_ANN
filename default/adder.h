/*
 * adder.h
 *
 *  Created on: Feb 26, 2018
 *      Author: keyan
 */

#ifndef ADDER_H_
#define ADDER_H_

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <time.h>
#include "ann_fixed.h"

#define RANGE 100
void init();
float add(int, int); 

void adder_O_sum(float sum);
int adder();
void adder_I_num1(int);
void adder_I_num2(int);
void adder_I_add();


#endif /* ADDER_H_ */
