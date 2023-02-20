
# CapsNet in C

## Overview
In 2019 my colleagues and I worked on accelerating the inference phase of CapsNet on an FPGA. During this project I wrote the CapsNet inference in C to serve as reference for lower level implementation. I created this repository to showcase this code. By no means is this implementation optimized for performance, however the availability of this implementation at such a low level can serve as a detailed reference of the exact operations going on behind the scenes.

## Running the code
*Important Disclaimer: The filepaths in this code are written in accordance to the UNIX convention with forward slashes (`/path/to/file`)*
### Building
In the repository directory:
```bash
mkdir build
cd build
cmake ..
make
```
### Executable
Building the code will result in an executable in the repository directory named `CapsNet`. This executable needs to be in the same directory as the `sample_input_image.txt` file and the `weights_txt` directory. This is the case by default.

## Files
### src
- `capsule_inference.c` contains the custom library functions
- `CapsNet_inference.c` contains the main function that utilizes these functions to create the model and run the inference
- `h5_format.c` handles the loading of the pretrained model weights from `weights_txt`
- `image_input_format.c` handles the loading of the sample input image in `sample_input_image.txt`

### Sample Input Image
The sample input image is a 28x28 grey-scale image of a hand-written digit 0 in txt format.

### Weights
There are sample weights from a model in the form of txt files in the directory `weights_txt`. These weights were extracted from a pretrained model for testing purposes.
