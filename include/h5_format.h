#ifndef H5_FORMAT
#define H5_FORMAT
#include "capsule_inference.h"

#define MAX_FN 50

void load_Conv(ConvLayer *layer, int layer_id);
void load_DigitCaps(DigitCapsLayer *layer, int layer_id);

#endif