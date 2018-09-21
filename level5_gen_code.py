import level4_k210

default_conv_arg = None
default_act_arg = None
default_bn_arg = {
    'load_para': 0,
    'bwsx_base_addr': 0
}
default_pool_arg = {
    'pool_type': 0,  # bypass
}



#
# def q_reverse(qvalue, ranges, mean):
#     return qvalue * ranges + mean
#
#
# def gen_layers_conv_weights(klayers: [level4_k210.K210Layer], x_range, x_mean, w_range, w_mean):
#         arg_add = x_mean * w_mean + w_mean * x_range
#         shr_w, arg_w = log_next_pow_of_2(w_range)
#         shr_x, arg_x = log_next_pow_of_2(x_range)


def gen_layer_struct(klayer: level4_k210.K210Layer, idx: int):
    todo = None
    reserved = 0
    set_to_zero = 0

    conv_arg = klayer.conv and klayer.conv.to_k210(idx) or default_conv_arg
    act_arg = klayer.act and klayer.act.to_k210() or default_act_arg
    bn_arg = klayer.bn and klayer.bn.to_k210(conv_arg['extra_scale']) or default_bn_arg
    pool_arg = klayer.pool and klayer.pool.to_k210() or default_pool_arg
    io_arg = klayer.to_k210()

    interrupt_enabe = {
        'int_en': todo,
        'ram_flag': reserved,
        'full_add': todo,
        'depth_wise_layer': conv_arg['depth_wise_layer']
    }
    image_addr = {
        'image_src_addr': todo,
        'image_dst_addr': todo
    }
    image_channel_num = {
        'i_ch_num': io_arg['i_ch_num'] - 1,
        'o_ch_num': io_arg['o_ch_num'] - 1,
        'o_ch_num_coef': io_arg['o_ch_num_coef'] - 1,
    }
    image_size = {
        'i_row_wid': conv_arg['i_row_wid'] - 1,
        'i_col_high': conv_arg['i_col_high'] - 1,
        'o_row_wid': io_arg['o_row_wid'] - 1,
        'o_col_high': io_arg['o_col_high'] - 1,
    }
    kernel_pool_type_cfg = {
        'kernel_type': conv_arg['kernel_type'],
        'pad_type': conv_arg['pad_type'],
        'pool_type': pool_arg['pool_type'],
        'first_stride': conv_arg['first_stride'],
        'bypass_conv': 0 if klayer.conv else 1,
        'load_para': bn_arg['load_para'],
        'dma_burst_size': io_arg['dma_burst_size'],
        'pad_value': conv_arg['pad_value'],
        'bwsx_base_addr': bn_arg['bwsx_base_addr'],
    }
    kernel_load_cfg = {
        'load_coor': conv_arg['load_coor'],
        'load_time': conv_arg['load_time'] - 1,
        'para_size': conv_arg['para_size'],
        'para_start_addr': None,
    }
    kernel_offset = {
        'coef_column_offset': set_to_zero,
        'coef_row_offset': set_to_zero,
    }
    kernel_calc_type_cfg = {
        'channel_switch_addr': conv_arg['channel_switch_addr'],
        'row_switch_addr': conv_arg['row_switch_addr'],
        'coef_size': reserved,
        'coef_group': conv_arg['coef_group'],
        'load_act': 1 if klayer.act else 0,
        'active_addr': act_arg['name']  # todo
    }
    write_back_cfg = {
        'wb_channel_switch_addr': io_arg['wb_channel_switch_addr'],
        'wb_row_switch_addr': io_arg['wb_row_switch_addr'],
        'wb_group': io_arg['wb_group']
    }
    conv_value = {
        'shr_w': conv_arg['shr_w'],
        'shr_x': conv_arg['shr_x'],
        'arg_w': hex(int(0x1000000+(conv_arg['arg_w']*(1<<23)))%0x1000000),
        'arg_x': hex(int(0x1000000+(conv_arg['arg_x']*(1<<23)))%0x1000000),
    }
    conv_value2 = {
        'arg_add': int(round((1<<40)*conv_arg['arg_add'])),
    }
    dma_parameter = {
        'send_data_out': io_arg['send_data_out'],
        'channel_byte_num': io_arg['channel_byte_num'] - 1,
        'dma_total_byte': io_arg['dma_total_byte'] - 1,
    }

    pass
    return {
        'interrupt_enabe': interrupt_enabe,
        'image_addr': image_addr,
        'image_channel_num': image_channel_num,
        'image_size': image_size,
        'kernel_pool_type_cfg': kernel_pool_type_cfg,
        'kernel_load_cfg': kernel_load_cfg,
        'kernel_offset': kernel_offset,
        'kernel_calc_type_cfg': kernel_calc_type_cfg,
        'write_back_cfg': write_back_cfg,
        'conv_value': conv_value,
        'conv_value2': conv_value2,
        'dma_parameter': dma_parameter
    }




def gen_layer_list_struct(klayers: [level4_k210.K210Layer]):
    ret = [
        gen_layer_struct(klayer, idx)
        for klayer, idx in zip(klayers, range(len(klayers)))
    ]
    return ret

def gen_layer_code(dlayer):
    return ('{\n' +
            ',\n'.join([
                ' .' + reg_name + '.data = {\n' +
                ',\n'.join([
                    '  .' + str(k) + ' = ' + str(v)
                    for k, v in data.items() if str(k) not in ('todo',)
                ]) + '\n }'
                for reg_name, data in dlayer.items()
            ]) +
            '\n}')


def gen_layer_list_code(klayers: [level4_k210.K210Layer]):
    return [
        gen_layer_code(layer)
        for layer in gen_layer_list_struct(klayers)
    ]
