import math

import level2_layers
import numpy as np


def log_next_pow_of_2(value):
    ret = 0
    while value > 1 or value <= -1:
        value = value / 2
        ret = ret + 1

    return ret, value


def pow_next_log_of_2(value, bound_shift, max_shift_shift=4):
    ret = 0
    max_shift = 1 << max_shift_shift
    while value >= -(1 << (bound_shift - 2)) and value < (1 << (bound_shift - 2)) \
            and value != 0 and ret < (max_shift - 1):
        value = value * 2
        ret = ret + 1

    return ret, value


def signed_to_hex(value, width):
    return hex(int((1 << width) + value) % (1 << width))


class K210Conv:
    def __init__(self, layer, sess, dataset, idx, input_min, input_max):
        self.layer = layer
        self.depth_wise_layer = isinstance(layer, level2_layers.LayerDepthwiseConvolutional)
        self.tensor = layer.tensor
        self.sess = sess
        self.dataset = dataset
        self.input_min = input_min
        self.input_max = input_max
        self.idx = idx

        self.x_range = None
        self.x_bias = None
        self.w_range = None
        self.w_mean = None
        self.output_shape = self.layer.tensor_conv_y.shape

    @staticmethod
    def q(value, ranges, mean):
        return (value - mean) / ranges

    def collection(self):
        batch_w = self.sess.run(self.layer.tensor_conv_w, self.dataset)
        ordered_w = np.sort(np.reshape(batch_w, [np.product(batch_w.shape)]))

        # # todo debug
        # batch_y = self.sess.run(self.layer.tensor_conv_y, self.dataset)
        # pass # debug

        # if self.idx != 0:
        #     batch_x = self.sess.run(self.layer.tensor_conv_x, self.dataset)
        #     ordered_x = np.sort(np.reshape(batch_x, [np.product(batch_x.shape)]))
        #     x_min = ordered_x[0]
        #     x_max = ordered_x[-1]
        #     self.input_min = x_min
        #     self.input_max = x_max

        self.x_range = self.input_max - self.input_min
        self.x_bias = self.input_min
        assert (self.x_range > 0)
        # w_min = ordered_w[int(len(ordered_w) * 0.05)]
        # w_max = ordered_w[int(len(ordered_w) * 0.95)]
        w_min = ordered_w[0]
        w_max = ordered_w[-1]
        self.w_range = w_max - w_min
        self.w_mean = w_min
        assert (self.w_range > 0)

    # @staticmethod
    # def weights_fill_buffer_33(weights, buf_size):
    #     reorder = [[[[weights[w][h][i_ch][o_ch]
    #                   for w in range(int(weights.shape[0]))]
    #                  for h in range(int(weights.shape[1]))]
    #                 for i_ch in range(int(weights.shape[2]))]
    #                for o_ch in range(int(weights.shape[3]))]
    #
    #     weights_o_ch_list = [
    #         np.array(o_ch_weights).flatten()
    #         for o_ch_weights in reorder
    #     ]
    #
    #     weights_shape = weights.shape
    #     weight_size = 2
    #     o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * int(weights_shape[2]) * weight_size
    #     n = math.floor(buf_size / o_ch_weights_size)
    #     return K210Layer.batch(weights_o_ch_list, n)
    #
    # @staticmethod
    # def weights_fill_buffer_11(weights, buf_size):
    #     reorder = [[[[weights[w][h][i_ch][o_ch]
    #                   for w in range(int(weights.shape[0]))]
    #                  for h in range(int(weights.shape[1]))]
    #                 for i_ch in range(int(weights.shape[2]))]
    #                for o_ch in range(int(weights.shape[3]))]
    #
    #     weights_o_ch_list = [
    #         [[*batch, None] for batch in K210Layer.batch(np.array(o_ch_weights).flatten(), 8)]
    #         for o_ch_weights in reorder
    #     ]
    #
    #     weights_shape = weights.shape
    #     weight_size = 2
    #     o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * int(weights_shape[2]) * weight_size
    #     n = math.floor(buf_size / o_ch_weights_size)
    #     return K210Layer.batch(weights_o_ch_list, n)

    def to_k210(self):
        self.collection()
        weight_buffer_size = 2 * 9 * 4096
        weight_q = np.transpose(self.q(self.layer.weights, self.w_range, self.w_mean), [3, 2, 0, 1]) * 65535
        weights = self.layer.weights

        input_shape = self.layer.tensor_conv_x.shape
        weights_shape = self.layer.tensor_conv_w.shape
        img_data_size = 1
        weight_data_size = 2
        img_line_size = 64
        img_memory_size = 1024 * 1024 * 2
        weight_cache_row_size = 9 * 2
        weight_cache_mem_size = weight_cache_row_size * 64

        input_row_size = int(input_shape[2]) * img_data_size
        input_channel_size = int(input_shape[1]) * input_row_size
        input_all_size = int(input_shape[3]) * input_channel_size
        output_row_size = int(input_shape[2]) * img_data_size
        output_channel_size = int(input_shape[1]) * output_row_size
        output_all_size = int(input_shape[3]) * output_channel_size
        kernel_size = int(weights_shape[0])
        weight_kernel_size = kernel_size * kernel_size * weight_data_size

        weight_all_size = weight_kernel_size * int(weights_shape[2]) * int(weights_shape[3])

        # exports:
        bypass_conv = 0
        # img i
        i_row_wid = int(input_shape[2])
        i_col_high = int(input_shape[1])
        coef_group = 1 if i_row_wid > 32 else (2 if i_row_wid > 16 else 4)
        row_switch_addr = math.ceil(i_row_wid / 64)
        channel_switch_addr = i_col_high * row_switch_addr #if coef_group == 1 else math.ceil(i_col_high / coef_group)
        # conv
        depth_wise_layer = 1 if self.depth_wise_layer else 0
        kernel_type = {1: 0, 3: 1}[kernel_size]
        pad_type = 0
        load_coor = 1
        weights_ich = int(weights_shape[2])
        weights_och = int(weights_shape[3])

        if kernel_size == 3:
            load_time = math.ceil(weights_och / math.floor(4096 / weights_ich))
        elif kernel_size == 1:
            load_time = math.ceil(weights_och / math.floor(4096 * 8 / weights_ich))
        else:
            assert (None)

        para_size = int(weight_all_size / load_time)

        para_start_addr = [int(round(item)) for item in np.reshape(weight_q, (np.product(weight_q.shape),))]
        first_stride = 0 if self.layer.config['stride'] == 1 else 1
        assert (256 > (i_col_high if first_stride == 0 else i_col_high / 2))

        # if idx == 0:
        #     bais_x, scale_x = (0, 1 / 256)
        # else:
        #     bais_x, scale_x = (self.x_bias, self.x_range / 256)
        bais_x, scale_x = (self.x_bias, self.x_range / 255)

        bais_w, scale_w = self.w_mean, self.w_range / (1 << (8 * weight_data_size))
        bx_div_sx = bais_x / scale_x
        bw_div_sw = bais_w / scale_w

        shr_x, arg_x = pow_next_log_of_2(bw_div_sw, 24)
        shr_w, arg_w = pow_next_log_of_2(bx_div_sx, 24)
        arg_add = kernel_size * kernel_size * bw_div_sw * bx_div_sx
        pad_value = -bx_div_sx
        swsx = scale_w * scale_x

        return locals()


class K210BN:
    def __init__(self, mean, var, gamma, beta):
        self.mean = mean
        self.var = var
        self.gamma = gamma
        self.beta = beta

    @staticmethod
    def get_bn(scale, bias):
        norm_shift, norm_mul = pow_next_log_of_2(scale, 24)
        return {'norm_mul': signed_to_hex(norm_mul, 24), 'norm_add': signed_to_hex(bias, 32), 'norm_shift': norm_shift}

    def to_k210(self, swsx=1):
        __tmp_hotfix_magic = 100000000.0 / 3
        scale = swsx * self.gamma / self.var * __tmp_hotfix_magic
        bias = (self.beta - self.gamma * self.mean / self.var) * __tmp_hotfix_magic

        # print('gamma', self.gamma, 'beta', self.beta, 'mean',self.mean, 'sigma', self.var, 'scale', scale, 'bias', bias)
        load_para = 1
        bwsx_base_addr = [
            self.get_bn(s, b)
            for s, b in zip(scale.tolist(), bias.tolist())
        ]

        return locals()


class K210Act:
    def __init__(self, layer, sess, dataset, name):
        self.layer = layer
        self.sess = sess
        self.dataset = dataset
        self.name = name
        self.min_y = None
        self.max_y = None

    @staticmethod
    def leaky_relu(x):
        return x if x >= 0 else 0.1 * x

    @staticmethod
    def leaky_relu_inverse(y):
        return y if y >= 0 else 10 * y

    @staticmethod
    def leaky_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 14 for i in range(14)]
        y_table.append(0)
        y_table.append(max_y)
        y_table = sorted(y_table)
        x_table = [K210Act.leaky_relu_inverse(it) for it in y_table]
        dydx = [(y_table[i + 1] - y_table[i]) / (x_table[i + 1] - x_table[i]) for i in range(len(y_table)-1)]
        return zip(x_table, y_table, dydx)

    @staticmethod
    def linear_table(min_y, max_y):
        range_y = max_y - min_y
        y_table = [min_y + i * range_y / 14 for i in range(14)]
        y_table.append(0)
        y_table.append(max_y)
        y_table = sorted(y_table)
        return zip(y_table, y_table, [1] * (len(y_table)-1))

    @staticmethod
    def find_shift(dydx):
        assert (dydx >= 0)
        ret_shift = 0
        while abs(dydx) < (1 << 14) and dydx != 0:
            dydx = dydx * 2
            ret_shift = ret_shift + 1
        return ret_shift, dydx

    @staticmethod
    def table_to_act(act_table, min_y, max_y):
        __tmp_hotfix_magic = 100000000.0 / 3
        act_table = [(0x800000000,0,0)]+[(x * __tmp_hotfix_magic, y, dydx / __tmp_hotfix_magic) for x, y, dydx in act_table]
        scale_y = 255 / (max_y - min_y)
        bias_y = -min_y * scale_y

        def ret_aux(x, y, dydx):
            dydx_scaled = dydx * scale_y
            y_scaled = round(y * scale_y + bias_y)
            dxss, dys = K210Act.find_shift(dydx_scaled)
            return {'x': round(x), 'y': y_scaled, 'dxs': dxss, 'dy': round(dys)}

        return [ret_aux(x, y, dydx) for x, y, dydx in act_table]

    def collection(self):
        batch_y = self.sess.run(self.layer.tensor_activation, self.dataset)
        ordered_y = np.sort(np.reshape(batch_y, [np.product(batch_y.shape)]))
        self.min_y = ordered_y[0]
        self.max_y = ordered_y[-1]

    def to_k210(self):
        self.collection()
        act_tab = None
        if self.name == 'leaky':
            act_tab = list(K210Act.leaky_table(self.min_y, self.max_y))[:16]
        elif self.name == 'linear':
            act_tab = list(K210Act.linear_table(self.min_y, self.max_y))[:16]
        else:
            assert (None)
        return {'active_addr': K210Act.table_to_act(list(act_tab), self.min_y, self.max_y)}


class K210Pool:
    def __init__(self, layer, name, size, stride, sess, dataset):
        self.name = name
        self.size = size
        self.stride = stride
        self.tensor = layer.tensor_pool
        self.sess = sess
        self.dataset = dataset




    def to_k210(self):
        # debug todo
        def q8(a, minv, maxv):
            scale = (maxv - minv) / 255
            bias = minv
            return (a - bias) / scale

        batch_y = self.sess.run(self.tensor, self.dataset)
        batch_x = self.sess.run(self.tensor.op.inputs[0], self.dataset)

        ordered_x = np.sort(np.reshape(batch_x, [np.product(batch_x.shape)]))
        minx = ordered_x[0]
        maxx = ordered_x[-1]
        np.set_printoptions(formatter={'int': hex})
        batch_y = q8(batch_y, minx, maxx).round().astype('int')
        batch_x = q8(batch_x, minx, maxx).round().astype('int')
        iy = batch_y[0].transpose([2,0,1])
        ix = batch_x[0].transpose([2,0,1])
        # print("=============", self.tensor.name)
        # with open('mid_data/'+self.tensor.name, 'w') as fout:
        #     K210Pool.debug_format(iy, fout)
        # print(iy[0][0][:16])
        pass # debug

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

    def to_k210(self, idx):
        if self.pool is not None:
            output_shape = self.pool.tensor.shape
        else:
            output_shape = self.conv.layer.tensor_conv_y.shape

        weights_shape = self.conv.layer.tensor_conv_w.shape
        input_shape = self.conv.layer.tensor_conv_x.shape
        i_row_wid = int(input_shape[1])
        weight_data_size = 2
        img_data_size = 1
        img_line_size = 64
        buf_size = 4096 * 3 * 3 * weight_data_size

        if self.conv.depth_wise_layer:
            o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * weight_data_size
        else:
            o_ch_weights_size = int(weights_shape[0]) * int(weights_shape[1]) * int(weights_shape[2]) * weight_data_size

        if int(weights_shape[0])==1:
            o_ch_weights_size = math.ceil(o_ch_weights_size/8)*9
        else:
            assert(int(weights_shape[0])==3)

        coef_group = 1 if i_row_wid > 32 else (2 if i_row_wid > 16 else 4)

        # io
        i_ch_num = int(weights_shape[2])
        o_ch_num = int(output_shape[3])
        o_ch_num_coef = min(math.floor(buf_size / o_ch_weights_size), int(output_shape[3]))
        # img o
        o_row_wid = int(output_shape[2])
        o_col_high = int(output_shape[1])
        wb_group = 1 if o_row_wid > 32 else (2 if o_row_wid > 16 else 4)
        wb_row_switch_addr = math.ceil(o_row_wid / 64)
        wb_channel_switch_addr = o_col_high * wb_row_switch_addr #if wb_group == 1 else math.ceil(o_col_high / wb_group)
        channel_byte_num = o_row_wid * o_col_high

        int_en = 0
        image_src_addr = None
        image_dst_addr = None
        dma_total_byte = o_row_wid * o_col_high * o_ch_num
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
        if len(ret) > 0:
            last_act = ret[-1].act
            last_act.collection()
            last_min = last_act.min_y
            last_max = last_act.max_y
        else:
            last_min = 0
            last_max = 1

        if isinstance(buffer[-1], level2_layers.LayerConvolutional) \
                or isinstance(buffer[-1], level2_layers.LayerDepthwiseConvolutional):
            conv_layer = buffer.pop()
            # assert (isinstance(conv_layer, level2_layers.LayerConvolutional))
            cur_k210.conv = K210Conv(conv_layer, sess, dataset, len(ret), last_min, last_max)
            if int(conv_layer.config['batch_normalize']) == 1:
                cur_k210.bn = K210BN(
                    conv_layer.batch_normalize_moving_mean,
                    conv_layer.batch_normalize_moving_variance,
                    conv_layer.batch_normalize_gamma,
                    conv_layer.batch_normalize_beta
                )
            else:
                bias_shape = conv_layer.bias.shape
                cur_k210.bn = K210BN(0, 1, np.ones(bias_shape), conv_layer.bias)

            cur_k210.act = K210Act(conv_layer, sess, dataset, conv_layer.config['activation'])

        if len(buffer) > 0 and isinstance(buffer[-1], level2_layers.LayerMaxpool):
            pool_layer = buffer.pop()
            assert (isinstance(pool_layer, level2_layers.LayerMaxpool))
            cur_k210.pool = K210Pool(pool_layer, 'maxpool', pool_layer.config['size'], pool_layer.config['stride'],
                                     sess, dataset)

        ret.append(cur_k210)

    return ret
