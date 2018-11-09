
## Dependent
you need `python3` `tensorflow` and `pillow`.
```sh
pip3 install tensorflow
pip3 install pillow
```

## Convert
clone the model-compiler. \
convert `.pb` file to k210
```sh
python3 model-compiler --pb_path <your pb file> --tensorboard-mode
```
check your inout tensor name and your output tensor name,
check your input dataset image width and height.
```sh
python3 model-compiler --pb_path <pb file path> --tensor_output_name <output tensor name> \
  --tensor_input_name <output input name> --dataset_input_name <input dataset tensor name> \
  --tensor_input_min <min value in input tensor> --tensor_input_max <max value in input tensor> \
  --image_w <image width> --image_h <image height> --dataset_pic_path <example image path> \
  --output_path <output path for compile result>
```

## Arguments
###-h
show more help
###--eight_bit_mode
using 8bit mode or 16bit mode

###--tesorboard-mode
run tensorboard for current `.pb` file
