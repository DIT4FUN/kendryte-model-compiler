import tensorflow as tf


class LayerBase:
    def __init__(self):
        self.name = 'no_name'
        self.config = {}

    def type_match(self, nodes, type_list):
        if len(nodes) != len(type_list):
            return False
        else:
            for node, ty in zip(nodes, type_list):
                if node.op.type != ty:
                    return False
        return True


class LayerNet(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.name = 'net'
        self.config = {'TODO': 'TODO'}


class LayerConvolutional(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.name = 'convolutional'
        self.config = {}
        batch_norm = None
        activation = None
        if self.type_match(info, ['BiasAdd', 'Conv2D']):
            biasadd, conv2d = info
        elif self.type_match(info, ['Relu', 'BiasAdd', 'Conv2D']):
            activation, biasadd, conv2d = info
        elif self.type_match(info, ['LeakyRelu', 'BiasAdd', 'Conv2D']):
            activation, biasadd, conv2d = info
        elif self.type_match(info, ['Relu6', 'BiasAdd', 'Conv2D']):
            activation, biasadd, conv2d = info
        elif self.type_match(info, ['Relu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, biasadd, conv2d = info
        elif self.type_match(info, ['LeakyRelu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, biasadd, conv2d = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, biasadd, conv2d = info
        else:
            print('not supported convolutional info.')
            return

        self.config['batch_normalize'] = 1 if batch_norm is not None else 0
        if batch_norm is not None:
            self.batch_normalize_mean = sess.run(batch_norm.op.inputs[1])
            self.batch_normalize_offset = sess.run(batch_norm.op.inputs[2])
            self.batch_normalize_a = sess.run(batch_norm.op.inputs[3])
            self.batch_normalize_b = sess.run(batch_norm.op.inputs[4])

        assert (isinstance(conv2d, tf.Tensor))
        self.config['size'] = int(conv2d.op.inputs[1].shape[0])
        self.config['stride'] = conv2d.op.get_attr('strides')[1]
        self.config['pad'] = 1 if conv2d.op.get_attr('padding') != 'SAME' else 0
        if activation is not None:
            self.config['activation'] = activation.op.type
        else:
            self.config['activation'] = 'linear'


class LayerDepthwiseConvolutional(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.name = 'depthwise_convolutional'
        self.config = {}
        if self.type_match(info, ['Relu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bais_add, dwconv = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bais_add, dwconv = info
        elif self.type_match(info, ['LeakyRelu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bais_add, dwconv = info
        else:
            print('not supported dw_convolutional info.')
            return

        self.config['batch_normalize'] = 1 if batch_norm is not None else 0
        if batch_norm is not None:
            self.batch_normalize_mean = sess.run(batch_norm.op.inputs[1])
            self.batch_normalize_offset = sess.run(batch_norm.op.inputs[2])
            self.batch_normalize_a = sess.run(batch_norm.op.inputs[3])
            self.batch_normalize_b = sess.run(batch_norm.op.inputs[4])

        assert (isinstance(dwconv, tf.Tensor))
        self.config['size'] = int(dwconv.op.inputs[1].shape[0])
        self.config['stride'] = dwconv.op.get_attr('strides')[1]
        self.config['pad'] = 1 if dwconv.op.get_attr('padding') != 'SAME' else 0
        if activation is not None:
            self.config['activation'] = activation.op.type
        else:
            self.config['activation'] = 'linear'


class LayerMaxpool(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.name = 'maxpool'
        self.config = {}
        if self.type_match(info, ['MaxPool']):
            max_pool = info[0]
        else:
            print('not supported maxpool info.')
            return

        assert (isinstance(max_pool, tf.Tensor))
        self.config['ksize'] = max_pool.op.get_attr('ksize')[1]
        self.config['stride'] = max_pool.op.get_attr('strides')[1]


def make_layer(sess, info):
    ty = info[0]
    info = info[1:]
    if ty == 'net':
        return LayerNet(sess, info)
    elif ty == 'convolutional':
        return LayerConvolutional(sess, info)
    elif ty == 'depthwise_convolutional':
        return LayerDepthwiseConvolutional(sess, info)
    elif ty == 'maxpool':
        return LayerMaxpool(sess, info)
    else:
        print('unknown type:', ty)


def make_layers(sess, info_list):
    info_list.reverse()
    return [make_layer(sess, info) for info in info_list]
