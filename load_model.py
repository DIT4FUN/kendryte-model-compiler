import tensorflow as tf
import numpy as np
from PIL import Image

from tensorflow.python.platform import gfile

import tensor_head_to_tensor_list
import tensor_list_to_layers
import layer_list_to_darknet
import layer_list_to_k210_layer
import k210_layer_to_c_code


def load_graph():
    with tf.Session() as persisted_sess:
        print("load graph")
        with gfile.FastGFile("graph_yv2_DW.pb", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')


        writer = tf.summary.FileWriter("./graphs", persisted_sess.graph)
        writer.close()

        return persisted_sess.graph._nodes_by_name['yv2'].outputs[0]


def box_image(im_path, new_h, new_w):
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


def main():
    t = load_graph()

    with tf.Session() as sess:
        converter = tensor_head_to_tensor_list.PbConverter(t)
        converter.convert()
        layers = tensor_list_to_layers.convert_to_layers(sess, converter.dst)
        dataset = np.array([box_image(path, 240, 320)[0].tolist() for path in
                            # ('pic/001.jpg', 'pic/002.jpg', 'pic/003.jpg', 'pic/004.jpg', 'pic/005.jpg', 'pic/006.jpg')
                            ('pic/dog.bmp', )
                            ])
        k210_layers = layer_list_to_k210_layer.gen_k210_layers(layers, sess, {'input:0': dataset})

        code = k210_layer_to_c_code.gen_layer_list_code(k210_layers)
        # print(level3_gen_file.gen_config_file(layers))
        # weights = level3_gen_file.gen_weights(layers)
        # print(len(weights))
        with open('gencode_output.c', 'w') as of:
            of.write(code)

        # print(code)
        pass


main()
