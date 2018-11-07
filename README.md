
## Dependent
you need `python3` `tensorflow` and `pillow`.
```sh
pip3 install tensorflow
pip3 install pillow
```

## Convert
convert `.pb` file to k210
```sh
git clone https://git.b-bug.org/maix/model-compiler.git
python3 model-compiler --pb_path <your pb file> --tensorboard-mode
```
check your inout tensor name and your output tensor name,
check your input dataset image width and height.
```sh
python3 model-compiler --pb_path <pb file path> --tensor_head_name <output tensor name>  \
  --dataset_input_name <input dataset tensor name> --image_w <image width> --image_h <image height> \
  --dataset_pic_path <example image path> --output_path <output path for compile result>
```

## Arguments
###-h
show more help
###--eight_bit_mode
using 8bit mode or 16bit mode

###--tesorboard-mode
run tensorboard for current `.pb` file
