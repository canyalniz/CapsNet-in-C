#ifndef CNN_INFERENCE
#define CNN_INFERENCE
#define EPSILON 1e-07
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct tensor *Tensor;
typedef struct tensor2D *Tensor2D;
typedef struct Kernel_Box KernelBox;
typedef struct Bias_Array BiasArray;
typedef struct Convolutional_Layer ConvLayer;
typedef struct Primary_Capsule_Layer PrimaryCapsLayer;
typedef struct Digit_Capsule_Layer DigitCapsLayer;

struct tensor{
    int dims[3]; //Dimensions
    float ***T; //Tensor itself
};


struct tensor2D{
    int dims[2]; //Dimensions
    float **T; //Tensor itself
};


struct Kernel_Box{ //Each kernel box holds a depth of kernels, i.e. in Conv1 we have 32 'kernel boxes', each 3x3x3
    int dims[3];
    float ***KB;
};


struct Bias_Array{
    int size;
    float *B;
};

struct Convolutional_Layer{
    int n_kb; //Number of kernel boxes in this layer
    int kernel_box_dims[3];
    KernelBox *kernel_box_group;
    BiasArray bias_array;
    int stride;
    int padding;
};

struct Primary_Capsule_Layer{
    int capsule_dimension; //dimension of output vector of each capsule
    ConvLayer *convolution_block; //the output of this single convolutional layer will be split to get the 8D capsule outputs
};

struct Digit_Capsule_Layer{
    int num_routings;
    int weight_dims[4]; // in our case 10x1152x16x8 (number of digit (output)capsules x number of primary (input)capsules x digit (output)capsule dimension x primary (input)capsule dimension)
    float ****weights;
};

//Layer generators
ConvLayer *empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride, int padding);
ConvLayer *new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array, int stride, int padding, int copy);
PrimaryCapsLayer *new_Primary_Caps(int capsule_dimension, ConvLayer *convolution_block);
DigitCapsLayer *emmpty_Digit_Caps(int num_routings, int num_caps_out, int num_caps_in, int dim_caps_out, int dim_caps_in);
DigitCapsLayer *new_Digit_Caps(int num_routings, int num_caps_out, int num_caps_in, int dim_caps_out, int dim_caps_in, float ****weights, int copy);

//Tensor operations
Tensor Conv(Tensor input, ConvLayer *layer, int dispose_of_input, Tensor (*activation)(Tensor,int));
Tensor2D PrimaryCaps(Tensor input, PrimaryCapsLayer *layer, int dispose_of_input);
Tensor2D DigitCaps(Tensor2D input, DigitCapsLayer *layer, int dispose_of_input);
Tensor2D squash_activation(Tensor2D input, int dispose_of_input);
Tensor ReLU_activation(Tensor input, int dispose_of_input);
Tensor linear_activation(Tensor input, int dispose_of_input);
Tensor apply_padding(Tensor input, int padding, int dispose_of_input);

//utility functions
void print_tensor(Tensor t);
void print_tensor2D(Tensor2D t);
float ****alloc_4D(int b, int d, int h, int w);
float ***alloc_3D(int d, int h, int w);
float **alloc_2D(int h, int w);
void print_conv_details(ConvLayer layer);
void free_tensor(Tensor t);
void free_tensor2D(Tensor2D t);
Tensor make_tensor(int d, int h, int w, float ***array);
Tensor2D make_tensor2D(int h, int w, float **array);

float *length_along_rows(Tensor2D tensor);

#endif