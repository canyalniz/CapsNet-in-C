#include "capsule_inference.h"
#include "h5_format.h"
#include "image_input_format.h"
#include <time.h>

int main(){
    int i;
    
    clock_t start, end;
    double cpu_time_used;
    
    // load input image
    char *fn = "sample_input_image.txt";
    float ***img;
    img = load_grayscale(fn, 28, 28);
    
    Tensor input;
    input = make_tensor(1, 28, 28, img);
    
    // load layers
    ConvLayer *conv1;
    conv1 = empty_Conv(256, 1, 9, 9, 1, 0);
    load_Conv(conv1, 1); 
    
    ConvLayer *convolution_block; 
    PrimaryCapsLayer *primary_capsules;
    convolution_block = empty_Conv(256, 256, 9, 9, 2, 0);
    load_Conv(convolution_block, 2);
    primary_capsules = new_Primary_Caps(8, convolution_block);
    
    DigitCapsLayer *digit_capsules;
    digit_capsules = emmpty_Digit_Caps(3, 10, 1152, 16, 8);
    load_DigitCaps(digit_capsules, 6);

    // inference
    Tensor x;
    Tensor2D t, output;

    start = clock(); // start clock
    x = Conv(input, conv1, 0, ReLU_activation);

    t = PrimaryCaps(x, primary_capsules, 0);

    output = DigitCaps(t, digit_capsules, 0);
    
    /*
    printf("\n\n"); 
    print_tensor2D(output);   YOU CAN USE THIS SNIPPET TO PRINT OUT THE OUTPUT OF DIGITCAPS DIRECTLY
    */

    float *lengths;
    lengths = length_along_rows(output);
    
    end = clock(); // stop clock
    cpu_time_used = ((double) (end - start)) / CLOCKS_PER_SEC;
    
    printf("\n\n");
    for(i=0; i<output->dims[0]; i++){
        printf("%d : %f\n", i, lengths[i]);
    }

    printf("\n\nInference completed in %f second(s).\n", cpu_time_used);

    return 0;
}