#include "h5_format.h"

void load_Conv(ConvLayer *layer, int layer_id){
    FILE *fp;
    char filename[MAX_FN];
    int n,d,h,w;
    for(h=0; h<layer->kernel_box_dims[1]; h++){
        for(w=0; w<layer->kernel_box_dims[2]; w++){
            sprintf(filename, "weights_txt/layer%d_h%d_w%d_weights.txt", layer_id, h, w);
            if((fp=fopen(filename, "r"))==NULL){
                fprintf(stderr, "Error, unable to access %s", filename);
                exit(EXIT_FAILURE);
            }
            
            for(d=0; d<layer->kernel_box_dims[0]; d++){
                for(n=0; n<layer->n_kb; n++){
                    fscanf(fp, "%f", &(layer->kernel_box_group[n].KB[d][h][w]));
                }
            }
            fclose(fp);
        }
    }

    sprintf(filename, "weights_txt/layer%d_biases.txt", layer_id);
    if((fp=fopen(filename, "r"))==NULL){
        fprintf(stderr, "Error, unable to access %s", filename);
        exit(EXIT_FAILURE);
    }

    for(n=0; n<layer->n_kb; n++){
        fscanf(fp, "%f", &(layer->bias_array.B[n]));
    }
    fclose(fp);
}

void load_DigitCaps(DigitCapsLayer *layer, int layer_id){
    FILE *fp;
    char filename[MAX_FN];
    sprintf(filename, "weights_txt/digit_caps_weights.txt");
    if((fp=fopen(filename, "r"))==NULL){
        fprintf(stderr, "Error, unable to access %s", filename);
        exit(EXIT_FAILURE);
    }
    int nco, nci, dco, dci;
    for(nco=0; nco<layer->weight_dims[0]; nco++){
        for(nci=0; nci<layer->weight_dims[1]; nci++){
            for(dco=0; dco<layer->weight_dims[2]; dco++){
                for(dci=0; dci<layer->weight_dims[3]; dci++){
                    fscanf(fp, "%f", &(layer->weights[nco][nci][dco][dci]));
                }
            }
        }
    }

    fclose(fp);
}