#include "main.h"

int adder();
void adder_I_num1(int);
void adder_I_num2(int);
void adder_I_add();

void init();

int main()
{
    srand(time(NULL));

	int layers[3] = {2, 2, 1};
	int activation[2] = {4, 2}; // 1 i tanh, 2 is ReLu
    ANN adder_ann;
    ann_init(&adder_ann, 3, layers, 1, activation);
    // adder_ann.activation[0] = 4;

    FILE * fp; 
    fp = fopen("adder.train", "w");

    int samples = 10000;
    fprintf(fp, "%d\n", samples);
    for(int i = 0; i < samples; i++)
    {
        int x = rand()%RANGE;
        int y = rand()%RANGE;

        float quotient = 2 * RANGE;

        fprintf(fp, "%f %f %f\n", ((float)x)/quotient, ((float)y)/quotient, ((float)(x + y))/quotient);
    }
    fclose(fp);

    ann_train_batch(&adder_ann, "adder.train", 1000, 0.0025, 1);

    ann_save_to_file(&adder_ann, "adder.net");

    printf("\nTesting ANN\n");

    FILE * rp;

    for(int j = 0; j < 3;j++)
    {        
        if(j == 0)
        {
            rp = fopen("result.csv", "w");
            adder_ann.activation[0] = 1;
        }
        else if(j == 1)
        {
            rp = fopen("result1.csv", "w");
            adder_ann.activation[0] = 4;
        }
        else
        {
            rp = fopen("result2.csv", "w");
            adder_ann.activation[0] = 5;
        }
        int x_count = 0;
        int y_count = 0;
        for(int i = 1; i <= 10000; i++)
        {
            if(x_count == 100)
            {
                x_count = 0;
                y_count++;
            }
            
            // int x = x_count;
            // int y = y_count;            
            
            // int y = rand()%RANGE;

            float quotient = 2 * RANGE;

            float inputs[2] = {((float)x_count)/quotient, ((float)y_count)/quotient};
            float outputs[1];
            float sum = ((float)(x_count + y_count));
            ann_run(inputs, outputs, &adder_ann);

            printf("%f + %f = %f (%f)\n", (float)x_count, (float)y_count, outputs[0] * quotient, (float)(x_count + y_count));
            fprintf(rp,"%d,%d,%f\n",x_count,y_count, outputs[0] * quotient);
            x_count++;   
            
        }
        fclose(rp);
    }

    // init();
    // int i;
    // for(i = 0; i < 10; i++)
    // {
    //     int x = rand()%RANGE;
    //     int y = rand()%RANGE;
    //     adder_I_num1(x);
    //     adder_I_num2(y);
    //     adder_I_add();
    //     adder();
    // }

    // return 0;
}

void adder_O_sum(float sum)
{
    printf("Answer: %f\n", sum);
}