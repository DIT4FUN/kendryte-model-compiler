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
        self.config = {}
        if self.type_match(info, ['Placeholder']):
            x, = info
        else:
            print('not supported net info.')
            return

        _, self.config['width'], self.config['height'], self.config['channels'] = x.shape.as_list()
        self.config['batch'] = 1
        self.config['subdivisions'] = 1


class LayerConvolutional(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.name = 'convolutional'
        self.config = {}
        batch_norm = None
        activation = None
        bais_add = None
        if self.type_match(info, ['BiasAdd', 'Conv2D']):
            biasadd, conv2d = info
        elif self.type_match(info, ['Relu', 'BiasAdd', 'Conv2D']):
            activation, bais_add, conv2d = info
        elif self.type_match(info, ['LeakyRelu', 'BiasAdd', 'Conv2D']):
            activation, bais_add, conv2d = info
        elif self.type_match(info, ['Relu6', 'BiasAdd', 'Conv2D']):
            activation, bais_add, conv2d = info
        elif self.type_match(info, ['Relu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, bais_add, conv2d = info
        elif self.type_match(info, ['LeakyRelu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, bais_add, conv2d = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, bais_add, conv2d = info
        else:
            print('not supported convolutional info.')
            return

        self.config['batch_normalize'] = 1 if batch_norm is not None else 0

        assert (isinstance(conv2d, tf.Tensor))
        self.config['size'] = int(conv2d.op.inputs[1].shape[0])
        self.config['stride'] = conv2d.op.get_attr('strides')[1]
        self.config['pad'] = 1 if conv2d.op.get_attr('padding') != 'SAME' else 0
        if activation is not None:
            self.config['activation'] = activation.op.type
        else:
            self.config['activation'] = 'linear'

        self.weights = sess.run(conv2d.op.inputs[1])
        if bais_add is not None:
            self.bias = sess.run(bais_add.op.inputs[1])

        if batch_norm is not None:
            self.batch_normalize_gamma = sess.run(batch_norm.op.inputs[1])
            self.batch_normalize_beta = sess.run(batch_norm.op.inputs[2])
            batch_norm_1 = batch_norm.op.outputs[1]
            batch_norm_2 = batch_norm.op.outputs[2]
            batch_normal_outputs = [
                op for k, op in sess.graph._nodes_by_name.items()
                if len(op.inputs) == 2 and op.inputs[1] in (batch_norm_1, batch_norm_2)
            ]
            mean_tensor = batch_normal_outputs[0].inputs[0]
            variance_tensor = batch_normal_outputs[1].inputs[0]
            assert('moving_mean/read' in mean_tensor.name)
            assert('moving_variance/read' in variance_tensor.name)
            self.batch_normalize_moving_mean = sess.run(mean_tensor)
            self.batch_normalize_moving_variance = sess.run(variance_tensor)


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

        assert (isinstance(dwconv, tf.Tensor))
        self.config['size'] = int(dwconv.op.inputs[1].shape[0])
        self.config['stride'] = dwconv.op.get_attr('strides')[1]
        self.config['pad'] = 1 if dwconv.op.get_attr('padding') != 'SAME' else 0
        if activation is not None:
            self.config['activation'] = activation.op.type
        else:
            self.config['activation'] = 'linear'

        self.weights = sess.run(dwconv.op.inputs[1])
        self.bias = sess.run(bais_add.op.inputs[1])

        if batch_norm is not None:
            self.batch_normalize_mean = sess.run(batch_norm.op.inputs[1])
            self.batch_normalize_offset = sess.run(batch_norm.op.inputs[2])
            batch_norm_1 = batch_norm.op.outputs[1]
            batch_norm_2 = batch_norm.op.outputs[2]
            batch_normal_outputs = [
                op for k, op in sess.graph._nodes_by_name.items()
                if len(op.inputs) == 2 and op.inputs[1] in (batch_norm_1, batch_norm_2)
            ]
            self.batch_normalize_moving_mean = sess.run(batch_normal_outputs[0].inputs[0])
            self.batch_normalize_moving_variance = sess.run(batch_normal_outputs[1].inputs[0])


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
        self.config['size'] = max_pool.op.get_attr('ksize')[1]
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
