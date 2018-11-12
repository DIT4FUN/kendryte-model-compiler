# How to Use Kendryte Model Compiler

Kendryte model compiler is used to transform your model file to a C file, which could run on kendryte K210.  *pb* and *h5* format models are supported for now.

[TOC]

## Preparation

**Note**: We suppose ubuntu 16.04 LTS is used.

### Install python3

```shell
$ sudo apt install python3
```

### Install pip3

```shell
$ sudo apt install python3-pip
```

### Install tensorflow

```shell
$ pip3 install tensorflow
```

### Install Pillow

```shell
$ pip3 install Pillow
```

### Install tensorboard

```shell
$ pip3 install tensorboard
```

## Command Line Parameters

``` --tensorboard_mode```: *True* to run tensorboard to visualize CNN, *False* to disable tensorboard (*False* default)

``` --pb_path ```: specify the path of *pb*-file or *h5*-file

```--tensor_output_name```: the name of tensor output operation

```--tensor_input_name```: the name of tensor input

```--dataset_input_name```: the name of the input of CNN

```--dataset_pic_path```: specify the path of an auxiliary picture from the dataset for transforming float point to fixed point 

```--image_w```: the width of the CNN input

```--image_h```: the height of the CNN input

```--eight_bit_mode```: *True* to enbale 8-bit quantization, *False* to enable 16-bit quantization (*False* default)

```--output_path```: the path of C output file (```build/gencode_output.c``` default)

## Example

Here we will take ```pb_files/20classes_yolo.pb``` as an example to show how to run it on Kendryte K210 finally.

### Model  Transformation

Type the following command on the command line from the model compiler root directory:

```shell
$ python3 __main__.py --pb_path pb_files/20classes_yolo.pb --tensor_output_name yv2 --dataset_input_name input:0 --dataset_pic_path dataset/yolo_240_320/dog.bmp 
--image_w 320 --image_h 240 --eight_bit_mode True
```

where 

* tensorboard is disabled;
* the default C output path is used.

Then we get the results on the terminal:

```shell
convert done.
[layer 0] Conv2d_0/max_pooling scale/bias: 0.05654605042700674 -1.5273598
[layer 1] leaky_0_2 scale/bias: 0.07839658400591681 -1.7567936
[layer 2] max_pooling scale/bias: 0.05823037764605354 -1.4459509
[layer 3] leaky_0_5 scale/bias: 0.06303120033413756 -1.7397203
[layer 4] max_pooling_1 scale/bias: 0.052160360298904716 -0.866366
[layer 5] leaky_0_8 scale/bias: 0.049140731961119406 -1.3931104
[layer 6] max_pooling_2 scale/bias: 0.040679183660768996 -0.9365978
[layer 7] leaky_0_11 scale/bias: 0.026200088800168504 -0.8973342
[layer 8] max_pooling_3 scale/bias: 0.028559230355655447 -0.5477671
[layer 9] leaky_0_14 scale/bias: 0.027155494689941405 -0.64771914
[layer 10] max_pool scale/bias: 0.022327593261120365 -0.50797397
[layer 11] leaky_0_17 scale/bias: 0.036412710302016316 -0.70867795
[layer 12] leaky_1_18 scale/bias: 0.023906670364679073 -0.54187393
[layer 13] leaky_1_19 scale/bias: 0.03958601858101639 -0.7297833
[layer 14] leaky_20 scale/bias: 0.03762062671137791 -0.8825908
[layer 15] yv2 scale/bias: 0.12349298514571844 -13.528209
```

Also, if you're not sure about the name of the input of CNN, the name of tensor output operation, the name of tensor input, the width of the CNN input or the height of the CNN input, enable the tensorboard to determine them as follows:

```shell
$ python3 __main__.py --pb_path pb_files/20classes_yolo.pb --tensorboard_mode True
```

### Modify *Demo*

Download ```kendryte-standalone-sdk```  (the current version is V0.5.2) and ```kendryte-standalone-demo``` from [Github](https://github.com/kendryte/kendryte-standalone-demo). 

Copy ```kendryte-standalone-demo/kpu``` to ```kendryte-standalone-sdk/src/```, and copy the above generated ```gencode_output.c``` from ```model-compiler/build/``` to ```kendryte-standalone-sdk/src/kpu/``` to replace the origin one.

**Regenerate ```region_layer_array.include```**

You will find a python script named ```gen_array.py``` in ```kendryte-standalone-sdk/src/kpu/``` . Replace the scale/bias pair's values at the beginning of the file with those at the end of ```gencode_output.c```. Then type the following in command line to regenerate ```region_layer_array.include```.

```shell
$ python3 gen_array.py
```

**Edit ```main.c```**

Possible modifications are as follows:

```c
/* main.c */
...
/* number of classes */
#define CLASS_NUMBER 20
...
/* number of anchor boxes */
#define ANCHOR_NUM	5
/* w, h pairs of anchor boxes */
float g_anchor[ANCHOR_NUM * 2] = {1.08, 1.19, 3.42, 4.41, 6.63, 11.38, 9.42, 5.11, 16.62, 10.52};
...
/* label name and color of 20 classes */
class_lable_t class_lable[CLASS_NUMBER] =
{
    {"aeroplane", GREEN},
    {"bicycle", GREEN},
    {"bird", GREEN},
    {"boat", GREEN},
    {"bottle", 0xF81F},
    {"bus", GREEN},
    {"car", GREEN},
    {"cat", GREEN},
    {"chair", 0xFD20},
    {"cow", GREEN},
    {"diningtable", GREEN},
    {"dog", GREEN},
    {"horse", GREEN},
    {"motorbike", GREEN},
    {"person", 0xF800},
    {"pottedplant", GREEN},
    {"sheep", GREEN},
    {"sofa", GREEN},
    {"train", GREEN},
    {"tvmonitor", 0xF9B6}
};
...
int main(void)
{
    ...
    /**
      * 320 - the width of display
      * 240 - the height of display
      * 0.5 - confidence thershold to remove low-confidence boxes
      * 0.2 - IOU threshold for NMS
      * ANCHOR_NUM - number of anchor boxes
      * g_anchor - w, h pairs of anchor boxes
      **/
    region_layer_init(&task, 320, 240, 0.5, 0.2, ANCHOR_NUM, g_anchor);
    ...
}
```

**Note**: the above modifications are applied to yolo_v2-based model, you have to make additional modifications or reimplement the parsing process if you have a different one.

### Build & Download

Build the modified *demo*:

```shell
$ cd kendryte-standalone-sdk
$ mkdir build && cd build
$ cmake .. -DPROJ=kpu -DTOOLCHAIN=/opt/riscv-toolchain/bin
$ make
```

You will get ```kpu.bin``` in ```kendryte-standalone-sdk/build```. 

Download ```kpu.bin``` to [KD233](https://shop302377334.taobao.com) with [K-Flash](https://kendryte.com/downloads/). Hoory! The *demo* is running!!

**Tips**: please refer to official [website](https://kendryte.com/downloads/) , [forum](https://forum.kendryte.com) and [GIT](https://github.com/kendryte) to get development resources for kendryte K210.





















