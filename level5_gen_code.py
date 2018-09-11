import level4_k210


def gen_layer_code(layers:K210Layer):
    todo = None
    interrupt_enabe = {
        'int_en': todo,
        'ram_flag': todo,
        'full_add': todo,
        'depth_wise_layer': todo
    }
    image_addr = {
        'image_src_addr': todo,
        'image_dst_addr': todo
    }
    image_channel_num = {
        'i_ch_num': todo,
        'o_ch_num': todo,
        'o_ch_num_coef': todo,
    }
    image_size = {
        'i_row_wid': todo,
        'i_col_high': todo,
        'o_row_wid': todo,
        'o_col_high': todo,
    }
    kernel_pool_type_cfg = {
        'kernel_type': todo,
        'pad_type': todo,
        'pool_type': todo,
        'first_stride': todo,
        'bypass_conv': todo,
        'load_para': todo,
        'dma_burst_size': todo,
        'pad_value': todo,
        'bwsx_base_addr': todo,
    }
    kernel_load_cfg = {
        'load_coor': todo,
        'load_time': todo,
        'para_size': todo,
        'para_start_addr': todo,
    }
    kernel_offset = {
        'coef_column_offset': todo,
        'coef_row_offset': todo,
    }
    kernel_calc_type_cfg = {
        'channel_switch_addr': todo,
        'row_switch_addr': todo,
        'coef_size': todo,
        'coef_group': todo,
        'load_act': todo,
        'active_addr': todo
    }
    write_back_cfg = {
        'wb_channel_switch_addr': todo,
        'wb_row_switch_addr': todo,
        'wb_group': todo
    }
    conv_value = {
        'shr_w': todo,
        'shr_x': todo,
        'arg_w': todo,
        'arg_x': todo,
    }
    conv_value2 = {
        'arg_add': todo,
    }
    dma_parameter = {
        'send_data_out': todo,
        'channel_byte_num': todo,
        'dma_total_byte': todo,
    }

    return {
        interrupt_enabe,
        image_addr,
        image_channel_num,
        image_size,
        kernel_pool_type_cfg,
        kernel_load_cfg,
        kernel_offset,
        kernel_calc_type_cfg,
        write_back_cfg,
        conv_value,
        conv_value2,
        dma_parameter
    }







def gen_layer_list_code(layers:[K210Layer]):
    for layer in layers:
        gen_layer_code(layer)