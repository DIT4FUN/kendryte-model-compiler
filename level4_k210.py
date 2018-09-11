import math

import level2_layers
import numpy as np


def log_next_pow_of_2(value):
    ret = 0
    while value > 1:
        value = value / 2
        ret = ret + 1

    while value < 0.5:
        value = value * 2
        ret = ret - 1

    return ret, value


class K210Conv:
    def __init__(self, layer, sess, dataset):
        self.layer = layer
        self.depth_wise_layer = isinstance(layer, level2_layers.LayerDepthwiseConvolutional)
        self.tensor = layer.tensor
        self.sess = sess
        self.dataset = dataset
        self.x_range = None
        self.x_mean = None
        self.w_range = None
        self.w_mean = None

    def collection(self):
        batch_x = self.sess.run(self.layer.tensor_conv_x, self.dataset)
        ordered_x = np.sort(np.reshape(batch_x, [np.product(batch_x.shape)]))
        batch_w = self.sess.run(self.layer.tensor_conv_w, self.dataset)
        ordered_w = np.sort(np.reshape(batch_w, [np.product(batch_w.shape)]))

        assert (len(ordered_x) > 10)
        assert (len(ordered_w) > 10)
        x_min = ordered_x[int(len(ordered_x) * 0.05)]
        x_max = ordered_x[int(len(ordered_x) * 0.95)]
        self.x_range = x_max - x_min
        self.x_mean = (x_min + x_max) / 2
        assert (self.x_range > 0)
        w_min = ordered_w[int(len(ordered_w) * 0.05)]
        w_max = ordered_w[int(len(ordered_w) * 0.95)]
        self.w_range = w_max - w_min
        self.w_mean = (w_min + w_max) / 2
        assert (self.w_range > 0)

    @staticmethod
    def q(value, ranges, mean):
        return (value - mean) / ranges

    @staticmethod
    def q_reverse(qvalue, ranges, mean):
        return qvalue * ranges + mean

    def to_k210(self):
        self.collection()
        weight_shape = self.layer.weights.shape[1:]
        weight_row_length = np.product(weight_shape[:-1])
        weight_length = np.product(weight_shape) * 2  # 16 bit
        weight_buffer_size = 144 * 1024  # 144k
        weight_q = self.q(self.layer.weights, self.w_range, self.w_mean)

        depth_wise_layer = 1 if self.depth_wise_layer else 0
        kernel_type = {1: 0, 3: 1}[int(self.layer.tensor_conv_w.shape[1])]
        pad_type = 0
        bypass_conv = 0
        pad_value = int(self.q_reverse(0, self.x_range, self.x_mean) * 256)
        load_coor = 1
        load_time = math.ceil(weight_length / weight_buffer_size)
        para_size = weight_buffer_size if kernel_type == 1 else weight_buffer_size * 9 / 8
        para_start_addr = weight_q
        shr_w, arg_w = log_next_pow_of_2(self.w_range)
        shr_x, arg_x = log_next_pow_of_2(self.x_range)
        arg_add = self.x_mean * self.w_mean + self.w_mean * self.x_mean / self.x_range

        return locals()


class K210BN:
    def __init__(self, mean, var, gamma, beta):
        self.mean = mean
        self.var = var
        self.gamma = gamma
        self.beta = beta

    def to_k210(self):
        scale = self.gamma / self.var
        norm_add = self.beta - self.gamma * self.mean / self.var
        norm_shift, norm_mul = log_next_pow_of_2(scale)

        load_para = 1
        bwsx_base_addr = [
            norm_mul, norm_add, norm_shift
        ]

        return locals()


class K210Act:
    def __init__(self, name):
        self.name = name

    def to_k210(self):
        return {'name': self.name}


class K210Pool:
    def __init__(self, name, size, stride):
        self.name = name
        self.size = size
        self.stride = stride

    def to_k210(self):
        if self.name == 'maxpool':
            return {'pool_type': {
                (2, 2): 1,
                (2, 1): 9
            }[(self.size, self.stride)]}
        else:
            return None


class K210Layer:
    def __init__(self):
        self.input_shape = None
        self.output_shape = None
        self.conv = None
        self.bn = None
        self.act = None
        self.pool = None

    def to_k210(self):
        input_shape = self.conv.layer.tensor_conv_x.shape
        output_shape = self.conv.layer.tensor_conv_y.shape


        int_en = 1
        image_src_addr = None
        image_dst_addr = None
        i_ch_num = int(self.conv.layer.tensor_conv_w.shape[2])
        o_ch_num = int(self.conv.layer.tensor_conv_w.shape[3])
        o_ch_num_coef
        i_row_wid = int(input_shape[1])
        i_col_high = int(input_shape[2])
        o_row_wid = int(output_shape[1])
        o_col_high = int(output_shape[2])
        first_stride
        dma_burst_size
        channel_switch_addr = int(input_shape[2]*input_shape[1]/64)
        row_switch_addr = int(input_shape[1]/64)
        coef_group
        wb_channel_switch_addr = int(output_shape[2]*output_shape[1]/64)
        wb_row_switch_addr = int(output_shape[1]/64)
        wb_group
        send_data_out
        channel_byte_num = int(output_shape[1]*output_shape[2])
        dma_total_byte = int(output_shape[1]*output_shape[2]*output_shape[3])
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
