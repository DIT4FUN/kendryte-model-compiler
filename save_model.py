import tensorflow as tf
import tensorflow.contrib.slim as slim
import build_node_tree
import level1_convert
import level2_layers
import level3_gen_file


def build_model(config_input_width, config_input_height, config_input_channel, config_batch_size):
    MODEL_NAME = 'TinyYOLOv2-Mobilenet'

    with tf.name_scope('placeholders'):
        x = tf.placeholder(tf.float32, shape=[None, config_input_width, config_input_height,
                                              config_input_channel], name='x')  # input

        # TODO: Trainner 里加boxes label和cell masks, 不在这里加哦

        # self.y = tf.placeholder(tf.float32, shape=[None, 10])  # output, [batch_size, x, y, w, h, c]
        y = tf.placeholder(tf.float32, shape=[None, 13, 13, 5, 5])  # output, [batch_size, x, y, w, h, c]

    def _depthwise_separable_conv(inputs,
                                  num_pwc_filters,
                                  sc=None,
                                  downsample=False):
        """ Helper function to build the depth-wise separable convolution layer.
        """
        _stride = 2 if downsample else 1

        # skip pointwise by setting num_outputs=None
        depthwise_conv = slim.separable_convolution2d(inputs,
                                                      num_outputs=None,
                                                      stride=_stride,
                                                      depth_multiplier=1,
                                                      kernel_size=[3, 3],
                                                      activation_fn=None)

        bn = slim.batch_norm(depthwise_conv, center= True, scale=True)

        # TODO: test on relu6 or back to lrelu
        act = tf.nn.relu6(bn)

        pointwise_conv = slim.convolution2d(act,
                                            num_pwc_filters,
                                            kernel_size=[1, 1],
                                            activation_fn=None)
        bn = slim.batch_norm(pointwise_conv, center= True, scale=True)

        act = tf.nn.relu6(bn)

        return act

    # network architecture
    # with tf.name_scope(MODEL_NAME):
    if True:
        # end_points_collection = sc.name + '_end_points'

        # with slim.arg_scope([slim.batch_norm],
        #                     is_training=is_training,
        #                     activation_fn=tf.nn.leaky_relu,
        #                     fused=True):

        # 普通卷积
        net = slim.convolution2d(inputs=x,
                                 num_outputs=16,
                                 kernel_size=(3, 3),
                                 stride=1,
                                 activation_fn=None)

        net = slim.batch_norm(net, center= True, scale=True)
        net = tf.nn.relu6(net)

        net = slim.max_pool2d(net,
                              kernel_size=2,
                              stride=2)

        # depthwise separable 卷积
        net = _depthwise_separable_conv(net, 32)
        net = slim.max_pool2d(net,
                              kernel_size=2,
                              stride=2)

        net = _depthwise_separable_conv(net, 64)
        net = slim.max_pool2d(net,
                              kernel_size=2,
                              stride=2)

        net = _depthwise_separable_conv(net, 128)
        net = slim.max_pool2d(net,
                              kernel_size=2,
                              stride=2)

        net = _depthwise_separable_conv(net, 256)
        net = slim.max_pool2d(net,
                              kernel_size=2,
                              stride=2)

        net = _depthwise_separable_conv(net, 512)
        net = slim.max_pool2d(net,
                              kernel_size=2,
                              stride=1,
                              padding='same')

        net = _depthwise_separable_conv(net, 1024)
        net = _depthwise_separable_conv(net, 1024)

        # classifier
        #
        net = slim.convolution2d(inputs=net,
                                 num_outputs=20,  # ANCHORS * (CLASSES + 5)
                                 kernel_size=(1, 1),
                                 stride=1,
                                 activation_fn=None)

        # net = slim.fully_connected(inputs=net,
        #                      num_outputs=100,
        #                      scope='fc_9')

        # net = tf.reshape(net, shape=(self.config.batch_size, 13, 13, 5, 4))

        # with tf.name_scope('reshape'):
        #     net = tf.reshape(net, [config_batch_size, -1])
        #     net = tf.reshape(net, shape=(config_batch_size, 13, 13, 5, 4))

        # import ipdb;
        # ipdb.set_trace()
        # net = slim.convolution(inputs=self.x,
        #                          num_outputs=100,
        #                          kernel_size=(1, 1),
        #                          stride=1,
        #                          activation_fn=slim.linear,
        #                          scope='conv_9')

        return net

    # self.build_train()


def main():
    with tf.Session() as sess:
        net = build_model(416, 416, 3, 5)

        with tf.name_scope('init'):
            init = tf.global_variables_initializer()
            sess.run(init)

        rx = sess.run(tf.random_normal([5, 416, 416, 3]))
        sess.run(net, {'placeholders/x:0': rx})

        # builder = tf.saved_model.builder.SavedModelBuilder('/tmp/log/tf/savedbuilder')
        # builder.add_meta_graph_and_variables(sess, ['tag_a'])
        # builder.save(True)

        writer = tf.summary.FileWriter("/tmp/log/tf", sess.graph)
        writer.close()

        #     print(sess.graph.get_all_collection_keys())
        #     op_list = sess.graph._nodes_by_id
        #     for i, op in op_list.items():
        #         print(op.name, op.type, [item.name for item in op.inputs], [item.shape for item in op.outputs])
        #
        # tree = build_node_tree.build_node_tree(sess.graph._nodes_by_name)
        # fltree = {}
        # build_node_tree.tree_flatten(fltree, tree)

        # for var in tf.trainable_variables():
        #     print('[trainable]', var.name, var.shape)
        #
        # for var in tf.all_variables():
        #     print('[all]', var.name, var.shape)

        converter = level1_convert.GraphConverter(net)
        converter.convert_all()
        layers = level2_layers.make_layers(sess, converter.dst)

        print(level3_gen_file.gen_config_file(layers))

    print(net.graph)


main()
