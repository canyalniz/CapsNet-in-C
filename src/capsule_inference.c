#include "capsule_inference.h"

ConvLayer *empty_Conv(int n_kb, int d_kb, int h_kb, int w_kb, int stride, int padding){
    ConvLayer *clp;
    clp = malloc(sizeof(ConvLayer)); //clp: Convolutional Layer Pointer
    if(clp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to clp in new_Conv.");
		exit(EXIT_FAILURE);
    }

    KernelBox *boxes = malloc(n_kb*sizeof(KernelBox));
    if(boxes==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to boxes in new_Conv.");
		exit(EXIT_FAILURE);
    }

    int n,d,h,w;
    for(n=0; n<n_kb; n++){
        boxes[n].dims[0] = d_kb; //Kernel Box Depth
        boxes[n].dims[1] = h_kb; //Kernel Box Height
        boxes[n].dims[2] = w_kb; //Kernel Box Width
        boxes[n].KB = alloc_3D(d_kb, h_kb, w_kb);
    }

    clp->kernel_box_group = boxes;
    clp->n_kb = n_kb;
    clp->bias_array.size = n_kb;
    clp->kernel_box_dims[0]=d_kb;
    clp->kernel_box_dims[1]=h_kb;
    clp->kernel_box_dims[2]=w_kb;
    //The number of biases in our layers is equal to the depth of the output tensor, which is equal to the number of kernel boxes in the layer.
    //i.e. in Conv1 we have 32 biases, one for each layer of the output tensor which is 32x31x31
    clp->stride = stride;
    clp->padding = padding;
    clp->bias_array.B = malloc(n_kb*sizeof(float));

    return clp;
}


ConvLayer *new_Conv(int n_kb, int d_kb, int h_kb, int w_kb, float **** weights_array, float * biases_array, int stride, int padding, int copy){
    ConvLayer *clp;
    clp = malloc(sizeof(ConvLayer)); //clp: Convolutional Layer Pointer
    if(clp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to clp in new_Conv.");
		exit(EXIT_FAILURE);
    }

    KernelBox *boxes = malloc(n_kb*sizeof(KernelBox));
    if(boxes==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to boxes in new_Conv.");
		exit(EXIT_FAILURE);
    }

    int n,d,h,w;
    for(n=0; n<n_kb; n++){
        boxes[n].dims[0] = d_kb; //Kernel Box Depth
        boxes[n].dims[1] = h_kb; //Kernel Box Height
        boxes[n].dims[2] = w_kb; //Kernel Box Width
        if(copy){
            boxes[n].KB = alloc_3D(d_kb, h_kb, w_kb);
            for(d=0; d<d_kb; d++){
                for(h=0; h<h_kb; h++){
                    for(w=0; w<w_kb; w++){
                        boxes[n].KB[d][h][w] = weights_array[n][d][h][w];
                    }
                }
            }
        } else {
            boxes[n].KB = weights_array[n];
        }
    }

    clp->kernel_box_group = boxes;
    clp->n_kb = n_kb;
    clp->bias_array.size = n_kb;
    clp->kernel_box_dims[0]=d_kb;
    clp->kernel_box_dims[1]=h_kb;
    clp->kernel_box_dims[2]=w_kb;
    //The number of biases in our layers is equal to the depth of the output tensor, which is equal to the number of kernel boxes in the layer.
    //i.e. in Conv1 we have 32 biases, one for each layer of the output tensor which is 32x31x31
    clp->stride = stride;
    clp->padding = padding;

    if(copy){
        clp->bias_array.B = malloc(n_kb*sizeof(float));
        for(n=0; n<n_kb; n++){
            clp->bias_array.B[n] = biases_array[n];
        }
    } else {
        clp->bias_array.B = biases_array;
    }

    return clp;
}


PrimaryCapsLayer *new_Primary_Caps(int capsule_dimension, ConvLayer *convolution_block){
    PrimaryCapsLayer *pclp;
    pclp = malloc(sizeof(PrimaryCapsLayer)); //pclp: Primary Capsule Layer Pointer
    if(pclp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to pcp in new_Primary_Caps.");
		exit(EXIT_FAILURE);
    }
    
    pclp->capsule_dimension = capsule_dimension;
    pclp->convolution_block = convolution_block;

    return pclp;
}

DigitCapsLayer *emmpty_Digit_Caps(int num_routings, int num_caps_out, int num_caps_in, int dim_caps_out, int dim_caps_in){
    DigitCapsLayer *dclp;
    dclp = malloc(sizeof(DigitCapsLayer)); //dclp: Digit Capsule Layer Pointer
    if(dclp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to pcp in new_Digit_Caps.");
		exit(EXIT_FAILURE);
    }
    
    if(num_routings==0){
        fprintf(stderr, "Error: num_routings must be greater than 0 in new_Digit_Caps.");
		exit(EXIT_FAILURE);
    }

    dclp->num_routings = num_routings;
    dclp->weight_dims[0] = num_caps_out;
    dclp->weight_dims[1] = num_caps_in;
    dclp->weight_dims[2] = dim_caps_out;
    dclp->weight_dims[3] = dim_caps_in;

    dclp->weights = alloc_4D(num_caps_out, num_caps_in, dim_caps_out, dim_caps_in);

    return dclp;
}

DigitCapsLayer *new_Digit_Caps(int num_routings, int num_caps_out, int num_caps_in, int dim_caps_out, int dim_caps_in, float ****weights, int copy){
    DigitCapsLayer *dclp;
    dclp = malloc(sizeof(DigitCapsLayer)); //dclp: Digit Capsule Layer Pointer
    if(dclp==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to pcp in new_Digit_Caps.");
		exit(EXIT_FAILURE);
    }
    
    if(num_routings==0){
        fprintf(stderr, "Error: num_routings must be greater than 0 in new_Digit_Caps.");
		exit(EXIT_FAILURE);
    }

    dclp->num_routings = num_routings;
    dclp->weight_dims[0] = num_caps_out;
    dclp->weight_dims[1] = num_caps_in;
    dclp->weight_dims[2] = dim_caps_out;
    dclp->weight_dims[3] = dim_caps_in;

    if(copy){
        dclp->weights = alloc_4D(num_caps_out, num_caps_in, dim_caps_out, dim_caps_in);
        int i,j,k,l;
        for(i=0; i<num_caps_out; i++){
            for(j=0; j<num_caps_in; j++){
                for(k=0; k<dim_caps_out; k++){
                    for(l=0; l<dim_caps_in; l++){
                        dclp->weights[i][j][k][l] = weights[i][j][k][l];
                    }
                }
            }
        }
    } else {
        dclp->weights = weights;
    }

    return dclp;
}

Tensor Conv(Tensor input, ConvLayer *layer, int dispose_of_input, Tensor (*activation)(Tensor,int)){
    if(input->dims[0]!=layer->kernel_box_dims[0]){
        fprintf(stderr, "Error: The depth of the kernel boxes in this layer(%d) and that of its input tensor(%d) must match", layer->kernel_box_dims[0], input->dims[0]);
        exit(EXIT_FAILURE);
    }

    if(layer->padding!=0){
        input = apply_padding(input,layer->padding,dispose_of_input);
    }

    int output_d = layer->n_kb;
    int output_w, output_h;
    output_h = ((input->dims[1] /*+ 2*layer->padding */ - layer->kernel_box_dims[1])/layer->stride)+1;
    output_w = ((input->dims[2] /*+ 2*layer->padding */ - layer->kernel_box_dims[2])/layer->stride)+1;
    //This is just the formula for getting the output height and width given the input dimensions, padding, kernel(filter) dimensions and stride
    //In our case output_h=output_w as we have square kernels(filters)
    
    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,id,by,bx,i,j;
    float result;

    // This thing goes over the output array and calculates each cell's value one by one
    for(d=0; d<output_d; d++){ //output depth
        for(h=0; h<output_h; h++){ //output height
            for(w=0; w<output_w; w++){ //output width
                result = 0; //this will hold the sum of the convolutions over each "channel" of the input tensor(the sum over its depth)
                for(id=0; id<input->dims[0]; id++){ //input depth
                    by = h*layer->stride; //"begin y" defines where the top edge of the kernel window is on the input layer
                    bx = w*layer->stride; //"begin x" defines where the left edge of the kernel window is on the input layer
                    for(i=0; i<(layer->kernel_box_dims[1]); i++){ //traverses the height of kernel window
                        for(j=0; j<(layer->kernel_box_dims[2]); j++){ //traverses the width of kernel window
                            result += input->T[id][by+i][bx+j] * layer->kernel_box_group[d].KB[id][i][j];
                        }
                    }
                }
                
                //Add the bias
                result += layer->bias_array.B[d];
                output_array[d][h][w] = result;
            }
        }
    }
    
    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    output = activation(output, 1);
    
    if(dispose_of_input) free_tensor(input);
    
    return output;
}

// squash acts on a single vector, this implementation applies squash individually to each vector in the array
// you can find the squash function in the paper, this is not an exact implementation of the mathematical function
// because there are some edge cases which break the program, so we use an approximation
Tensor2D squash_activation(Tensor2D input, int dispose_of_input){
    Tensor2D output;
    int h,w;
    
    if(dispose_of_input){
        output = input;
    } else {
        float **output_array = alloc_2D(input->dims[0], input->dims[1]);
        output = make_tensor2D(input->dims[0], input->dims[1], output_array);
    }

    float **squared_array = alloc_2D(input->dims[0], input->dims[1]);
    float *squared_norm = calloc(input->dims[0],sizeof(float));
    float *scale_array =  malloc(input->dims[0]*sizeof(float));

    for(h=0; h<input->dims[0]; h++){
        for(w=0; w<input->dims[1]; w++){
            squared_array[h][w] = input->T[h][w] * input->T[h][w];
        }
    }

    for(h=0; h<input->dims[0]; h++){
        for(w=0; w<input->dims[1]; w++){
            squared_norm[h] += squared_array[h][w];
        }
        scale_array[h] = (squared_norm[h] / (1+squared_norm[h])) / sqrtf(squared_norm[h] + EPSILON); // we basically scale the vector in this activation function
    }

    for(h=0; h<input->dims[0]; h++){
        for(w=0; w<input->dims[1]; w++){
            output->T[h][w] = input->T[h][w] * scale_array[h];
        }
    }

    free(squared_norm);
    free(scale_array);
    for(h=0; h<input->dims[0]; h++){
        free(squared_array[h]);
    }
    free(squared_array);

    return output;
}

Tensor2D PrimaryCaps(Tensor input, PrimaryCapsLayer *layer, int dispose_of_input){
    Tensor result_buffer;
    result_buffer = malloc(sizeof(struct tensor));
    result_buffer = Conv(input, layer->convolution_block, dispose_of_input, linear_activation);

    int output_w, output_h;
    output_w = layer->capsule_dimension;
    output_h = result_buffer->dims[0]*result_buffer->dims[1]*result_buffer->dims[2]/output_w;
    
    float **output_array = alloc_2D(output_h,output_w);

    int ibd, ibh, ibw, oh, ow;
    
    
    // A reshape operation to arrange the (256x6x6) output tensor of the "convolution block" into (8D) vector outputs of (1152) capsules
    oh=0; // output height index
    for(ibh=0; ibh<result_buffer->dims[1]; ibh++){ // input begin height index
        for(ibw=0; ibw<result_buffer->dims[2]; ibw++){ // input begin width index
            for(ibd=0; ibd<result_buffer->dims[0]; ibd+=layer->capsule_dimension){ // input begin depth index
                for(ow=0; ow<layer->capsule_dimension; ow++){ // output width index
                    output_array[oh][ow] = result_buffer->T[ibd+ow][ibh][ibw];
                }
                oh++;
            }
        }
    }
    
    Tensor2D output;
    output = make_tensor2D(output_h, output_w, output_array);
    
    if(dispose_of_input) free_tensor(input);
    
    return squash_activation(output, dispose_of_input);
}

Tensor2D DigitCaps(Tensor2D input, DigitCapsLayer *layer, int dispose_of_input){
    int output_h = layer->weight_dims[0], output_w = layer->weight_dims[2];

    float **output_array = alloc_2D(output_h, output_w);
    Tensor2D output = make_tensor2D(output_h, output_w, output_array);
    float ***buffer = alloc_3D(output_h, input->dims[0], output_w);
    float **coupling_b = alloc_2D(input->dims[0], output_h); // there is one b value for each pair of input-output capsules, this matrix is initialized to 0 for each image during inference
    float **coupling_c = alloc_2D(input->dims[0], output_h); // the c values are obtained by applying softmax to coupling_b along the rows

    int oh, ow, ih, iw, r;
    float exp_sum, sum_buffer;

    // initialize coupling_b
    for(ih=0; ih<input->dims[0]; ih++){
        for(oh=0; oh<output_h; oh++){
            coupling_b[ih][oh] = 0;
        }
    }
    
    for(oh=0; oh<output_h; oh++){
        for(ow=0; ow<output_w; ow++){
            output->T[oh][ow] = 0;
        }
    }

    // for each output capsule get the "affine transformation matrix" of each input capsule and do the matrix multiplication
    for(oh=0; oh<output_h; oh++){ // output height, number of output capsules
        for(ih=0; ih<input->dims[0]; ih++){ // input height, number of input capsules
            for(ow=0; ow<output_w; ow++){ //output width, dimension of output capsules
                buffer[oh][ih][ow] = 0; // preperation for the dot product
                for(iw=0; iw<input->dims[1]; iw++){ // input width, dimension of input capsules / dot product between input (8D)capsule vector and respective affine transformation matrix row
                    buffer[oh][ih][ow] += layer->weights[oh][ih][ow][iw] * input->T[ih][iw];
                }
            }
        }

    }
    
    // DYNAMIC ROUTING
    for(r=0; r<layer->num_routings; r++){ // number of routings
        // apply softmax to coupling_b along its rows in order to get coupling_c
        for(ih=0; ih<input->dims[0]; ih++){
            exp_sum = 0;
            for(oh=0; oh<output_h; oh++){
                exp_sum += expf(coupling_b[ih][oh]);
            }

            for(oh=0; oh<output_h; oh++){
                coupling_c[ih][oh] = expf(coupling_b[ih][oh]) / exp_sum;
            }
        }

        // calculate output with current coupling coefficients
        for(oh=0; oh<output_h; oh++){
            for(ow=0; ow<output_w; ow++){
                sum_buffer = 0;
                for(ih=0; ih<input->dims[0]; ih++){
                    sum_buffer += buffer[oh][ih][ow] * coupling_c[ih][oh];
                }

                output->T[oh][ow] = sum_buffer;
            }
        }
        output = squash_activation(output, dispose_of_input);
        
        // in all but the last iteration we update the coupling coefficients
        if(r<(layer->num_routings-1)){
            
            // this loop calculates the dot product of the current output and the predicted output to see how much they agree
            for(ih=0; ih<input->dims[0]; ih++){ 
                for(oh=0; oh<output_h; oh++){
                    sum_buffer = 0;
                    for(ow=0; ow<output_w; ow++){
                        sum_buffer += buffer[oh][ih][ow] * output->T[oh][ow];
                    }
                    coupling_b[ih][oh] += sum_buffer;
                }
            }
        }
    }
    
    if(dispose_of_input) free_tensor2D(input);

    return output;
}

Tensor ReLU_activation(Tensor input, int dispose_of_input){
    Tensor output;
    int d,h,w;

    if(dispose_of_input){
        output = input;
    } else {
        float ***output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    for(d=0; d<output->dims[0]; d++){
        for(h=0; h<output->dims[1]; h++){
            for(w=0; w<output->dims[2]; w++){
                output->T[d][h][w] = (input->T[d][h][w] < 0) ? 0 : input->T[d][h][w];
            }
        }
    }

    return output;
}

Tensor linear_activation(Tensor input, int dispose_of_input){
    Tensor output;
    int d,h,w;

    if(dispose_of_input){
        output = input;
    } else {
        float ***output_array = alloc_3D(input->dims[0], input->dims[1], input->dims[2]);
        output = make_tensor(input->dims[0], input->dims[1], input->dims[2], output_array);
    }

    for(d=0; d<output->dims[0]; d++){
        for(h=0; h<output->dims[1]; h++){
            for(w=0; w<output->dims[2]; w++){
                output->T[d][h][w] = input->T[d][h][w];
            }
        }
    }

    return output;
}

Tensor apply_padding(Tensor input, int padding, int dispose_of_input){
    int output_d = input->dims[0];
    int output_h = input->dims[1] + 2*padding;
    int output_w = input->dims[2] + 2*padding;

    float ***output_array = alloc_3D(output_d,output_h,output_w);

    int d,h,w,x,y;
    
    for(d=0; d<output_d; d++){
        //pad top and bottom
        for(x=0; x<output_w; x++){
            output_array[d][0][x] = output_array[d][output_h-1][x] = 0;
        }
        //pad left and right
        for(y=0; y<output_h; y++){
            output_array[d][y][0] = output_array[d][y][output_w-1] = 0;
        }
        //load the middle
        for(x=padding; x<(output_w-padding); x++){
            for(y=padding; y<(output_h-padding); y++){
                output_array[d][y][x] = input->T[d][y-padding][x-padding];
            }    
        }
    }

    Tensor output;
    output = make_tensor(output_d, output_h, output_w, output_array);

    if(dispose_of_input) free_tensor(input);

    return output;
}


void print_tensor(Tensor t){
    int i,j,k;
    for(i=0; i<t->dims[0]; i++){
        printf("\nLayer %d:\n\n", i);
        for(j=0; j<t->dims[1]; j++){
            for(k=0; k<t->dims[2]; k++){
                printf("%f ", t->T[i][j][k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

void print_tensor2D(Tensor2D t){
    int i,j;
    for(i=0; i<t->dims[0]; i++){
        for(j=0; j<t->dims[1]; j++){
            printf("%f ", t->T[i][j]);
        }
        printf("\n");
    }
}

float ****alloc_4D(int b, int d, int h, int w){
    float **** new;
    new = malloc(b*sizeof(float***));
    if(new==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to new in alloc_4D.");
		exit(EXIT_FAILURE);
    }

    int i,j,k;
    for(i=0; i<b; i++){
        new[i] = malloc(d*sizeof(float**));
        if(new[i]==NULL){
            fprintf(stderr, "Error: Unable to allocate memory to new[%d] in alloc_4D.",i);
            exit(EXIT_FAILURE);
        }
        for(j=0; j<d; j++){
            new[i][j] = malloc(h*sizeof(float*));
            if(new[i][j]==NULL){
                fprintf(stderr, "Error: Unable to allocate memory to new[%d][%d] in alloc_4D.",i,j);
                exit(EXIT_FAILURE);
            }
            for(k=0; k<h; k++){
                new[i][j][k] = malloc(w*sizeof(float));
                if(new[i][j][k]==NULL){
                    fprintf(stderr, "Error: Unable to allocate memory to new[%d][%d][%d] in alloc_4D.",i,j,k);
                    exit(EXIT_FAILURE);
                }
            }
        }
    }
    return new;
}

float ***alloc_3D(int d, int h, int w){
    float ***new;
    new = malloc(d*sizeof(float**));
    if(new==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to new in alloc_3D.");
		exit(EXIT_FAILURE);
    }

    int i,j;
    for(i=0; i<d; i++){
        new[i] = malloc(h*sizeof(float*));
        if(new[i]==NULL){
            fprintf(stderr, "Error: Unable to allocate memory to new[%d] in alloc_3D.",i);
            exit(EXIT_FAILURE);
        }
        for(j=0; j<h; j++){
            new[i][j] = malloc(w*sizeof(float));
            if(new[i][j]==NULL){
                fprintf(stderr, "Error: Unable to allocate memory to new[%d][%d] in alloc_3D.",i,j);
                exit(EXIT_FAILURE);
            }
        }
    }
    return new;
}

float **alloc_2D(int h, int w){
    float **new;
    new = malloc(h*sizeof(float*));
    if(new==NULL){
        fprintf(stderr, "Error: Unable to allocate memory to new in alloc_2D.");
		exit(EXIT_FAILURE);
    }

    int i;
    for(i=0; i<h; i++){
        new[i] = malloc(w*sizeof(float));
        if(new[i]==NULL){
            fprintf(stderr, "Error: Unable to allocate memory to new[%d] in alloc_2D.",i);
            exit(EXIT_FAILURE);
        }
    }
    return new;
}

void print_conv_details(ConvLayer layer){
    printf("Convolutional layer at %x\n\n", &layer);
    printf("\tn_kb = %d\n", layer.n_kb);
    printf("\tkernel_box_dims = %d,%d,%d\n", layer.kernel_box_dims[0], layer.kernel_box_dims[1], layer.kernel_box_dims[2]);
    printf("\tstride = %d\n", layer.stride);
    printf("\tpadding = %d\n\n", layer.padding);

    int n,d,h,w;
    for(n=0; n<layer.n_kb; n++){
        printf("\tBox %d:\n", n);
        for(d=0; d<layer.kernel_box_group[n].dims[0]; d++){
            printf("\t\tLayer %d:\n", d);
            for(h=0; h<layer.kernel_box_group[n].dims[1]; h++){
                for(w=0; w<layer.kernel_box_group[n].dims[2]; w++){
                    printf("\t\t\t%f ", layer.kernel_box_group[n].KB[d][h][w]);
                }
                printf("\n");
            }
        }
    }
}

void free_tensor(Tensor t){
    int d,h;
    for(d=0; d<t->dims[0]; d++){
        for(h=0; h<t->dims[1]; h++){
            free(t->T[d][h]);
        }
        free(t->T[d]);
    }
    free(t->dims);
    //free(t);
}

void free_tensor2D(Tensor2D t){
    int h;
    for(h=0; h<t->dims[1]; h++){
        free(t->T[h]);
    }
    free(t->dims);
    //free(t);
}

Tensor make_tensor(int d, int h, int w, float ***array){
    Tensor new_tensor;
    new_tensor = malloc(sizeof(struct tensor));
    new_tensor->T = array;
    new_tensor->dims[0] = d;
    new_tensor->dims[1] = h;
    new_tensor->dims[2] = w;
    return new_tensor;
}

Tensor2D make_tensor2D(int h, int w, float **array){
    Tensor2D new_tensor;
    new_tensor = malloc(sizeof(struct tensor2D));
    new_tensor->T = array;
    new_tensor->dims[0] = h;
    new_tensor->dims[1] = w;
    return new_tensor;
}

float *length_along_rows(Tensor2D tensor){
    float *length;
    length = calloc(tensor->dims[0], sizeof(float));
    int i,j;
    for(i=0; i<tensor->dims[0]; i++){
        for(j=0; j<tensor->dims[1]; j++){
            length[i] += tensor->T[i][j] * tensor->T[i][j];
        }
        length[i] += EPSILON;
        length[i] = sqrtf(length[i]);
    }
    return length;
}