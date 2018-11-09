'''
 * Copyright 2018 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

import argparse
import os
import tempfile

import tensorflow as tf
import numpy as np

from tensorflow.python.platform import gfile

import tensor_head_to_tensor_list
import tensor_list_to_layer_list
import layer_list_to_k210_layer
import k210_layer_to_c_code

current_dir = os.path.dirname(os.path.realpath(__file__))

def load_graph(pb_file_path, tensor_output_name, tensor_input_name):
    if pb_file_path.endswith('h5'):
        import h5_converter
        pb_file_path = h5_converter.convert(pb_file_path)

    if pb_file_path.endswith('pb'):
        with tf.Session() as persisted_sess:
            with gfile.FastGFile(pb_file_path, 'rb') as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())
                persisted_sess.graph.as_default()
                tf.import_graph_def(graph_def, name='')

        output_tensor, input_tensor = None, None
        if tensor_output_name is not None:
            output_tensor = persisted_sess.graph._nodes_by_name[tensor_output_name].outputs[0]
        if tensor_input_name is not None:
            input_tensor = persisted_sess.graph._nodes_by_name[tensor_input_name].outputs[0]

        return output_tensor, input_tensor

    return None

def box_image(im_path, new_w, new_h):
    from PIL import Image
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

    ch_size = {'RGB':3}.get(orig.mode, 1)
    resized = np.array(orig.resize([n_w, n_h]), dtype='float32') / 255.0
    resized = resized.reshape([*resized.shape, ch_size][:3])

    box_im = np.ones([new_h, new_w, ch_size], dtype='float32') * 0.5
    fill_y = (new_h - n_h) >> 1
    fill_x = (new_w - n_w) >> 1
    box_im[fill_y:fill_y + n_h, fill_x:fill_x + n_w, :] = resized

    return box_im, resized

def convert(tensor_output, tensor_input, dataset_pack, eight_bit_mode=False, input_min=0, input_max=1):
    with tf.Session() as sess:
        converter = tensor_head_to_tensor_list.PbConverter(tensor_output, tensor_input)
        converter.convert()
        layers = tensor_list_to_layer_list.convert_to_layers(sess, converter.dst)
        k210_layers = layer_list_to_k210_layer.gen_k210_layers(
            layers, sess, dataset_pack,
            eight_bit_mode=eight_bit_mode,
            input_min=input_min,
            input_max=input_max
        )

        code = k210_layer_to_c_code.gen_layer_list_code(k210_layers, eight_bit_mode)
        return code



def main():
    def str2bool(v):
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise argparse.ArgumentTypeError('Boolean value expected.')

    parser = argparse.ArgumentParser()
    parser.add_argument('--tensorboard_mode', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--pb_path', type=str, default='<please set --pb_path>')
    parser.add_argument('--tensor_input_name', default=None)
    parser.add_argument('--tensor_output_name', default=None)
    parser.add_argument('--tensor_input_min', type=float, default=0)
    parser.add_argument('--tensor_input_max', type=float, default=1)
    parser.add_argument('--dataset_input_name', default='input:0')
    parser.add_argument('--dataset_pic_path', default='pic/yolo')
    parser.add_argument('--image_w', type=int, default=320)
    parser.add_argument('--image_h', type=int, default=240)
    parser.add_argument('--eight_bit_mode', type=str2bool, nargs='?', const=True, default=False)
    parser.add_argument('--output_path', default='build/gencode_output.c')

    # Deprecated
    parser.add_argument('--tensor_head_name', default=None)

    args = parser.parse_args()

    if args.tensor_head_name is not None:
        print(
            '[warning]: --tensor_head_name is deprecated. please use --tensor_output_name instand'
        )

    tensorboard_mode = args.tensorboard_mode
    pb_path = args.pb_path
    tensor_input_name = args.tensor_input_name
    tensor_output_name = args.tensor_output_name or args.tensor_head_name
    input_min = args.tensor_input_min
    input_max = args.tensor_input_max
    dataset_input_name = args.dataset_input_name
    dataset_pic_path = args.dataset_pic_path
    image_w = args.image_w
    image_h = args.image_h
    eight_bit_mode = args.eight_bit_mode
    output_path = args.output_path

    if ':' not in dataset_input_name:
        dataset_input_name = dataset_input_name + ':0'

    if tensorboard_mode:
        load_graph(pb_path, None, None)
        graphs_path = tempfile.mkdtemp('graphs')
        writer = tf.summary.FileWriter(graphs_path, tf.Session().graph)
        writer.close()
        import subprocess
        subprocess.call(['tensorboard', '--logdir', graphs_path])
        return


    tensor_output, tensor_input = load_graph(pb_path, tensor_output_name, tensor_input_name)
    if os.path.isdir(dataset_pic_path):
        import random
        all_files = os.listdir(dataset_pic_path)
        all_files = random.sample(all_files, min(128, len(all_files))) # set maxmum dataset size
        dataset_file_list = [
            os.path.join(dataset_pic_path, f)
            for f in all_files
            if os.path.isfile(os.path.join(dataset_pic_path, f))
        ]
    else:
        dataset_file_list = (dataset_pic_path, )

    dataset = np.array([box_image(path, image_w, image_h)[0].tolist() for path in dataset_file_list])

    code = convert(
        tensor_output, tensor_input,
        {dataset_input_name: dataset},
        eight_bit_mode=eight_bit_mode,
        input_min=input_min,
        input_max=input_max
    )

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w') as of:
            of.write(code)

if __name__ == '__main__':
    main()
