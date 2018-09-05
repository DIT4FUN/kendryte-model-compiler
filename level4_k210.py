import level2_layers
import numpy as np


class K210Conv:
    def __init__(self, layer, sess, dataset):
        self.layer = layer
        self.depth_wise_layer = isinstance(layer, level2_layers.LayerDepthwiseConvolutional)
        self.tensor = layer.tensor
        self.sess = sess
        self.dataset = dataset

    def collection(self):
        batch_x = self.sess.run(self.layer.tensor_conv_x, self.dataset)
        ordered_x = np.sort(np.reshape(batch_x, [np.product(batch_x.shape)]))
        batch_w = self.sess.run(self.layer.tensor_conv_w, self.dataset)
        ordered_w = np.sort(np.reshape(batch_w, [np.product(batch_w.shape)]))
        batch_y = self.sess.run(self.layer.tensor_conv_y, self.dataset)
        ordered_y = np.sort(np.reshape(batch_y, [np.product(batch_y.shape)]))
        pass

    def to_k210(self):
        self.collection()
        weight = self.layer.weights

        depth_wise_layer = 1 if self.depth_wise_layer else 0
        kernel_type = {1: 0, 3: 1}[int(self.tensor.shape[1])]
        pad_type = 0  # todo: check what for pad
        bypass_conv = 0
        pad_value = 0
        load_coor = 1
        # load_time
        # para_size
        # para_start_addr
        # shr_w
        # shr_x
        # arg_w
        # arg_x
        # arg_add

        return locals()


class K210BN:
    def __init__(self, mean, var, gamma, beta):
        self.mean = mean
        self.var = var
        self.gamma = gamma
        self.beta = beta

    def to_k210(self):
        pass


class K210Act:
    def __init__(self, name):
        self.name = name

    def to_k210(self):
        pass


class K210Pool:
    def __init__(self, name, size, stride):
        self.name = name
        self.size = size
        self.stride = stride

    def to_k210(self):
        pass


class K210Layer:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.conv = None
        self.bn = None
        self.act = None
        self.pool = None

    def to_k210(self):
        return self.conv and self.conv.to_k210()


def gen_k210_layers(layers: [level2_layers.LayerBase], sess, dataset):
    buffer = list(layers)
    buffer.reverse()
    ret = []

    net = buffer.pop()
    assert (isinstance(net, level2_layers.LayerNet))
    current_shape = int(net.config['width']), int(net.config['height']), int(net.config['channels'])

    while len(buffer) != 0:
        cur_k210 = K210Layer
        cur_k210.input_shape = buffer[-1].tensor[0].shape

        if isinstance(buffer[-1], level2_layers.LayerConvolutional) \
                or isinstance(buffer[-1], level2_layers.LayerDepthwiseConvolutional):
            conv_layer = buffer.pop()
            # assert (isinstance(conv_layer, level2_layers.LayerConvolutional))
            cur_k210.conv = K210Conv(conv_layer, sess, dataset)
            if int(conv_layer.config['batch_normalize']) == 1:
                cur_k210.bn = K210BN(
                    conv_layer.batch_normalize_moving_mean,
                    conv_layer.batch_normalize_moving_variance,
                    conv_layer.batch_normalize_gamma,
                    conv_layer.batch_normalize_beta
                )
            cur_k210.act = K210Act(conv_layer.config['activation'])

        if len(buffer) > 0 and isinstance(buffer[-1], level2_layers.LayerMaxpool):
            pool_layer = buffer.pop()
            assert (isinstance(pool_layer, level2_layers.LayerMaxpool))
            cur_k210.pool = K210Pool('maxpool', pool_layer.config['size'], pool_layer.config['stride'])

        ret.append(cur_k210)

    return ret
