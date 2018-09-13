import math

import level2_layers
import numpy as np


def log_next_pow_of_2(value):
    ret = 0
    while value > 1:
        value = value / 2
        ret = ret + 1

    # while value < 0.5:
    #     value = value * 2
    #     ret = ret - 1

    return ret, value


def log_next_pow_of_2_list(value_list):
    return [log_next_pow_of_2(value) for value in value_list]


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
        self.output_shape = self.layer.tensor_conv_y.shape

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

    @staticmethod
    def weights_fill_buffer_33(weights, buf_size):
        reorder = [[[[weights[w][h][i_ch][o_ch]
                      for w in range(int(weights.shape[0]))]
                     for h in range(int(weights.shape[1]))]
                    for i_ch in range(int(weights.shape[2]))]
                   for o_ch in range(int(weights.shape[3]))]

        weights_o_ch_list = [
            np.array(o_ch_weights).flatten()
            for o_ch_weights in reorder
        ]

        weights_shape = weights.shape
        weight_size = 2
        o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * int(weights_shape[2]) * weight_size
        n = math.floor(buf_size / o_ch_weights_size)
        return K210Layer.batch(weights_o_ch_list, n)

    @staticmethod
    def weights_fill_buffer_11(weights, buf_size):
        reorder = [[[[weights[w][h][i_ch][o_ch]
                      for w in range(int(weights.shape[0]))]
                     for h in range(int(weights.shape[1]))]
                    for i_ch in range(int(weights.shape[2]))]
                   for o_ch in range(int(weights.shape[3]))]

        weights_o_ch_list = [
            [[*batch, None] for batch in K210Layer.batch(np.array(o_ch_weights).flatten(), 8)]
            for o_ch_weights in reorder
        ]

        weights_shape = weights.shape
        weight_size = 2
        o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * int(weights_shape[2]) * weight_size
        n = math.floor(buf_size / o_ch_weights_size)
        return K210Layer.batch(weights_o_ch_list, n)

    def to_k210(self):
        self.collection()
        weight_shape = self.layer.weights.shape[1:]
        weight_buffer_size = 2*9*4096
        weight_q = self.q(self.layer.weights, self.w_range, self.w_mean)

        input_shape = self.layer.tensor_conv_x.shape
        output_shape = self.layer.tensor_conv_y.shape
        weights_shape = self.layer.tensor_conv_w.shape
        img_data_size = 1
        weight_data_size = 2
        img_line_size = 64
        img_memory_size = 1024 * 1024 * 2
        weight_cache_row_size = 9 * 2
        weight_cache_mem_size = weight_cache_row_size * 64

        input_row_size = int(input_shape[1]) * img_data_size
        input_channel_size = int(input_shape[2]) * input_row_size
        input_all_size = int(input_shape[3]) * input_channel_size
        output_row_size = int(input_shape[1]) * img_data_size
        output_channel_size = int(input_shape[2]) * output_row_size
        output_all_size = int(input_shape[3]) * output_channel_size
        kernel_size = int(weights_shape[0])
        weight_kernel_size = kernel_size * kernel_size * weight_data_size
        if kernel_size == 1:
            weight_single_output_size = math.ceil(int(weights_shape[0] * weights_shape[1]) / 8) * 9 * weight_data_size
        elif kernel_size == 3:
            weight_single_output_size = weight_kernel_size * int(weights_shape[2])
        else:
            raise "unsupport kernel_size: " + str(kernel_size)

        weight_all_size = weight_single_output_size * int(weights_shape[3])

        buf_size = 4096 * 3 * 3 * weight_data_size
        o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * int(weights_shape[2]) * weight_data_size

        # exports:
        # weights
        i_ch_num = int(weights_shape[2])
        o_ch_num = int(weights_shape[3])
        o_ch_num_coef = min(math.floor(buf_size / o_ch_weights_size), int(output_shape[3]))
        # img i
        i_row_wid = int(input_shape[1])
        i_col_high = int(input_shape[2])
        coef_group = 1 if i_row_wid > 32 else (2 if i_row_wid > 16 else 4)
        row_switch_addr = img_line_size * coef_group / 64
        channel_switch_addr = math.ceil(img_data_size * i_row_wid / img_line_size)
        # img o
        o_row_wid = int(output_shape[1])
        o_col_high = int(output_shape[2])
        wb_group = 1 if o_row_wid > 32 else (2 if o_row_wid > 16 else 4)
        wb_channel_switch_addr = img_line_size * coef_group
        wb_row_switch_addr = math.ceil(img_data_size * o_row_wid / img_line_size)
        channel_byte_num = wb_row_switch_addr * int(output_shape[3])
        # conv
        depth_wise_layer = 1 if self.depth_wise_layer else 0
        kernel_type = {1: 0, 3: 1}[int(self.layer.tensor_conv_w.shape[1])]
        pad_type = 0
        bypass_conv = 0
        pad_value = int(self.q_reverse(0, self.x_range, self.x_mean) * 256)
        load_coor = 1
        load_time = math.ceil(weight_all_size / weight_buffer_size)
        para_size = min(math.floor(weight_buffer_size / weight_single_output_size) * weight_single_output_size, weight_all_size)
        para_start_addr = weight_q
        shr_w, arg_w = log_next_pow_of_2(self.w_range)
        shr_x, arg_x = log_next_pow_of_2(self.x_range)
        arg_add = self.x_mean * self.w_mean + self.w_mean * self.x_mean / self.x_range
        first_stride = self.layer.config['stride']
        assert (256 > i_col_high if first_stride == 0 else i_col_high / 2)

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

        load_para = 1
        bwsx_base_addr = [
            {'norm_mul': norm_mul, 'norm_add': norm_add, 'norm_shift': norm_shift}
            for norm_shift, norm_mul in log_next_pow_of_2_list(scale)
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
        self.conv = None
        self.bn = None
        self.act = None
        self.pool = None

    @staticmethod
    def batch(iter, n=1):
        l = len(iter)
        for ndx in range(0, l, n):
            yield iter[ndx:min(ndx + n, l)]

    def to_k210(self):
        int_en = 0
        image_src_addr = None
        image_dst_addr = None
        dma_total_byte = int(np.product(self.conv.output_shape[1:]))
        dma_burst_size = 0xf
        send_data_out = 0
        return locals()


def gen_k210_layers(layers: [level2_layers.LayerBase], sess, dataset):
    buffer = list(layers)
    buffer.reverse()
    ret = []

    net = buffer.pop()
    assert (isinstance(net, level2_layers.LayerNet))
    current_shape = int(net.config['width']), int(net.config['height']), int(net.config['channels'])

    while len(buffer) != 0:
        cur_k210 = K210Layer()
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
