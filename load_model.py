import tensorflow as tf

from tensorflow.python.framework.graph_util import convert_variables_to_constants
from tensorflow.python.platform import gfile

import level1_convert
import level2_layers
import level3_gen_file


def load_graph():

    with tf.Session() as persisted_sess:
        print("load graph")
        with gfile.FastGFile("graph_yv2.graph", 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())
            persisted_sess.graph.as_default()
            tf.import_graph_def(graph_def, name='')
        # print("map variables")
        # persisted_result = persisted_sess.graph.get_tensor_by_name("saved_result:0")
        # tf.add_to_collection(tf.GraphKeys.VARIABLES, persisted_result)
        # try:
        #     saver = tf.train.Saver(tf.all_variables())  # 'Saver' misnomer! Better: Persister!
        # except:
        #     pass
        # print("load data")
        # saver.restore(persisted_sess, "checkpoint.data")  # now OK
        # print(persisted_result.eval())
        # print("DONE")

        writer = tf.summary.FileWriter("./graphs", persisted_sess.graph)
        writer.close()

        converter = level1_convert.GraphConverter(persisted_sess.graph._nodes_by_name['leaky_relu_8/Maximum'].outputs[0])
        converter.convert_all()
        layers = level2_layers.make_layers(persisted_sess, converter.dst)

        print(level3_gen_file.gen_config_file(layers))
        weights = level3_gen_file.gen_weights(layers)
        print(len(weights))


load_graph()
