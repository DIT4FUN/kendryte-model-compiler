import argparse

import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.python.platform import gfile

import tensor_head_to_tensor_list
import tensor_list_to_layer_list
import layer_list_to_k210_layer
import k210_layer_to_c_code


def load_graph(pb_file_path, tensor_head_name):
    with tf.Session() as persisted_sess:
        print("load graph")
        with gfile.FastGFile(pb_file_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')


        writer = tf.summary.FileWriter("./graphs", persisted_sess.graph)
        writer.close()

        return persisted_sess.graph._nodes_by_name[tensor_head_name].outputs[0]


def box_image(im_path, new_w, new_h):
    orig = Image.open(im_path)
    w, h = orig.size
    w_scale = float(new_w) / w
    h_scale = float(new_h) / h

    n_w = new_w
    n_h = new_h
    if w_scale < h_scale:
        n_h = int(h * w_scale)
    else:
        n_w = int(w * h_scale)

    resized = np.array(orig.resize([n_w, n_h]), dtype='float32') / 255.0

    box_im = np.ones([new_h, new_w, 3], dtype='float32') * 0.5
    fill_y = (new_h - n_h) >> 1
    fill_x = (new_w - n_w) >> 1
    box_im[fill_y:fill_y + n_h, fill_x:fill_x + n_w, :] = resized

    return box_im, resized

def convert(tensor_head, dataset_pack, eight_bit_mode=False):
    with tf.Session() as sess:
        converter = tensor_head_to_tensor_list.PbConverter(tensor_head)
        converter.convert()
        layers = tensor_list_to_layer_list.convert_to_layers(sess, converter.dst)
        k210_layers = layer_list_to_k210_layer.gen_k210_layers(layers, sess, dataset_pack, eight_bit_mode)

        code = k210_layer_to_c_code.gen_layer_list_code(k210_layers, eight_bit_mode)
        return code


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pb_path', type=str, default='pb_files/graph_yv2_DW.pb')
    parser.add_argument('--tensor_head_name', default='yv2')
    parser.add_argument('--dataset_input_name', default='input:0')
    parser.add_argument('--dataset_pic_path', default='pic/dog.bmp')
    parser.add_argument('--image_w', type=int, default=320)
    parser.add_argument('--image_h', type=int, default=240)
    parser.add_argument('--eight_bit_mode', type=bool, default=False)
    parser.add_argument('--output_path', default='build/gencode_output.c')
    args = parser.parse_args()

    pb_path = args.pb_path
    tensor_head_name = args.tensor_head_name
    dataset_input_name = args.dataset_input_name
    dataset_pic_path = args.dataset_pic_path
    image_w = args.image_w
    image_h = args.image_h
    eight_bit_mode = args.eight_bit_mode
    output_path = args.output_path


    tensor_head = load_graph(pb_path, tensor_head_name)
    dataset = np.array([box_image(path, image_w, image_h)[0].tolist() for path in
                        # ('pic/001.jpg', 'pic/002.jpg', 'pic/003.jpg', 'pic/004.jpg', 'pic/005.jpg', 'pic/006.jpg')
                        (dataset_pic_path, )
                        ])

    code = convert(tensor_head, {dataset_input_name: dataset}, eight_bit_mode)

    with open(output_path, 'w') as of:
            of.write(code)

main()
