'''
 * Copyright 2018 Canaan Inc.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 '''

import layer_list_to_k210_layer
import numpy as np
import math

default_conv_arg = None
default_act_arg = None
default_bn_arg = {
    'load_para': 0,
    'bwsx_base_addr': 0
}
default_pool_arg = {
    'pool_type': 0,  # bypass
}


def q(value, ranges, mean):
    return (value - mean) / ranges

def signed_to_hex(value, width):
    return hex(int(round((1 << width) + value)) % (1 << width))


def debug_format_line(line, fout):
    line = [*line, *([0] * (64 - len(line)))]
    ret = ''.join([format(v, '02x') + ('  ' if i % 8 == 7 else ('--' if i % 8 == 3 else '')) for v, i in
                   zip(line, range(len(line)))])
    fout.write('Address 0X00000000: ' + ret + '\n')


def split_chunks(L, n):
    for i in range(0, len(L), n):
        yield L[i:i + n]


def min_max_to_scale_bias(minv, maxv):
    scale = (maxv - minv) / 255
    bias = minv
    return scale, bias



def gen_layer_struct(klayer: layer_list_to_k210_layer.K210Layer, idx: int):
    reserved = 0
    set_to_zero = 0
    img_ram_size = 2 * 1024 * 1024

    conv_arg = klayer.conv and klayer.conv.to_k210() or default_conv_arg
    act_arg = klayer.act and klayer.act.to_k210() or default_act_arg
    bn_arg = klayer.bn and klayer.bn.to_k210(conv_arg['swsx']) or default_bn_arg
    pool_arg = klayer.pool and klayer.pool.to_k210() or default_pool_arg
    io_arg = klayer.to_k210(idx)

    if klayer.pool:
        tensor_out = klayer.pool.tensor
        tensor_pre_out = klayer.pool.tensor.op.inputs[0]
        batch_x = klayer.conv.sess.run(tensor_pre_out, klayer.conv.dataset)
        ordered_x = np.sort(np.reshape(batch_x, [np.product(batch_x.shape)]))
        batch_y = klayer.conv.sess.run(tensor_out, klayer.conv.dataset)
        mino = ordered_x[0]
        maxo = ordered_x[-1]
    else:
        tensor_out = klayer.act.layer.tensor_activation
        batch_y = klayer.conv.sess.run(tensor_out, klayer.conv.dataset)
        ordered_o = np.sort(np.reshape(batch_y, [np.product(batch_y.shape)]))
        mino = ordered_o[0]
        maxo = ordered_o[-1]

    output_scale, output_bias = min_max_to_scale_bias(mino, maxo)
    print("[layer {}]".format(idx), tensor_out.op.name, 'scale/bias:', output_scale, output_bias)


    img_input_size = int(math.ceil(io_arg['i_ch_num'] / conv_arg['coef_group']) * 64 * conv_arg['channel_switch_addr'])
    img_output_size = int(math.ceil(io_arg['o_ch_num'] / io_arg['wb_group']) * 64 * io_arg['wb_channel_switch_addr'])

    assert (img_input_size + img_output_size <= img_ram_size)

    interrupt_enabe = {
        'int_en': set_to_zero,
        'ram_flag': reserved,
        'full_add': set_to_zero,
        'depth_wise_layer': conv_arg['depth_wise_layer']
    }
    image_addr = {
        'image_src_addr': '(uint64_t)' + hex(int((0 if not idx & 1 else (img_ram_size - img_input_size)) / 64)),
        'image_dst_addr': '(uint64_t)' + hex(int((0 if idx & 1 else (img_ram_size - img_output_size)) / 64))
    }
    image_channel_num = {
        'i_ch_num': hex(io_arg['i_ch_num'] - 1),
        'o_ch_num': hex(io_arg['o_ch_num'] - 1),
        'o_ch_num_coef': hex(conv_arg['o_ch_num_coef'] - 1),
    }
    image_size = {
        'i_row_wid': hex(conv_arg['i_row_wid'] - 1),
        'i_col_high': hex(conv_arg['i_col_high'] - 1),
        'o_row_wid': hex(io_arg['o_row_wid'] - 1),
        'o_col_high': hex(io_arg['o_col_high'] - 1),
    }
    kernel_pool_type_cfg = {
        'kernel_type': conv_arg['kernel_type'],
        'pad_type': conv_arg['pad_type'],
        'pool_type': pool_arg['pool_type'],
        'first_stride': conv_arg['first_stride'],
        'bypass_conv': 0 if klayer.conv else 1,
        'load_para': bn_arg['load_para'],
        'dma_burst_size': io_arg['dma_burst_size'],
        'pad_value': signed_to_hex(conv_arg['pad_value'], 8),
        'bwsx_base_addr': bn_arg['bwsx_base_addr'],
    }
    kernel_load_cfg = {
        'load_coor': conv_arg['load_coor'],
        'load_time': conv_arg['load_time'] - 1,
        'para_size': conv_arg['para_size'],
        'para_start_addr': conv_arg['para_start_addr'],
    }
    kernel_offset = {
        'coef_column_offset': set_to_zero,
        'coef_row_offset': set_to_zero,
    }
    kernel_calc_type_cfg = {
        'channel_switch_addr': hex(conv_arg['channel_switch_addr']),
        'row_switch_addr': hex(conv_arg['row_switch_addr']),
        'coef_size': reserved,
        'coef_group': conv_arg['coef_group'],
        'load_act': 1 if klayer.act else 0,
        'active_addr': act_arg['active_addr']
    }
    write_back_cfg = {
        'wb_channel_switch_addr': hex(io_arg['wb_channel_switch_addr']),
        'wb_row_switch_addr': hex(io_arg['wb_row_switch_addr']),
        'wb_group': io_arg['wb_group']
    }
    conv_value = {
        'shr_w': conv_arg['shr_w'],
        'shr_x': conv_arg['shr_x'],
        'arg_w': signed_to_hex(conv_arg['arg_w'], 24),
        'arg_x': signed_to_hex(conv_arg['arg_x'], 24),
    }
    conv_value2 = {
        'arg_add': int(round(conv_arg['arg_add'])),
    }
    dma_parameter = {
        'send_data_out': io_arg['send_data_out'],
        'channel_byte_num': io_arg['channel_byte_num'] - 1,
        'dma_total_byte': io_arg['dma_total_byte'] - 1,
    }

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
    }, (output_scale, output_bias)


def gen_layer_list_struct(klayers: [layer_list_to_k210_layer.K210Layer]):
    ret = [
        gen_layer_struct(klayer, idx)
        for klayer, idx in zip(klayers, range(len(klayers)))
    ]
    return ret


def gen_layer_code(dlayer, idx):
    return ('// ' + str(idx) + '\n{\n' +
            ',\n'.join([
                ' .' + reg_name + '.data = {\n' +
                ',\n'.join([
                    '  .' + str(k) + ' = ' + (
                        str(v)
                        if str(k) not in ('bwsx_base_addr', 'para_start_addr', 'active_addr')
                        else '0'
                    )
                    for k, v in data.items()
                ]) + '\n }'
                for reg_name, data in dlayer[0].items()
            ]) +
            '\n}')


def gen_bn_code(dlayer, idx):
    bn_list = dlayer[0]['kernel_pool_type_cfg']['bwsx_base_addr']
    bn_code_list = [(' {.batchnorm.data = {' +
                     '.norm_mul = ' + str(bn['norm_mul']) + ', ' +
                     '.norm_add = ' + str(bn['norm_add']) + ', ' +
                     '.norm_shift = ' + str(bn['norm_shift']) +
                     '}}') for bn in bn_list]
    return 'kpu_batchnorm_argument_t bwsx_base_addr_' + str(
        idx) + '[] __attribute__((aligned(128))) = {\n' + ',\n'.join(bn_code_list) + '\n};'


def gen_act_code(dlayer, idx):
    act_list = dlayer[0]['kernel_calc_type_cfg']['active_addr']
    active_para = ' .activate_para = {\n' + ',\n'.join([
        '  {{.data = {{.shift_number={dxs}, .y_mul={dy}, .x_start={x} }}}}'.format(
            dxs=item['dxs'], dy=int(item['dy']), x=signed_to_hex(item['x'], 36)
        )
        for item in act_list
    ]) + '\n }'
    bias_list = [int(item['y']) for item in act_list]
    active_para_bias0 = (
                ' .activate_para_bias0.data = {{\n  .result_bias = {{{},{},{},{},{},{},{},{}}}\n }}'
    ).format(*(bias_list[:8]))

    active_para_bias1 = (
            ' .activate_para_bias1.data = {{\n  .result_bias = {{{},{},{},{},{},{},{},{}}}\n }}'
    ).format(*(bias_list[8:]))

    return 'kpu_activate_table_t active_addr_' + str(idx) + ' __attribute__((aligned(128))) = {\n' + \
           ',\n'.join([active_para, active_para_bias0, active_para_bias1]) + \
           '\n};'


def gen_weights_code(dlayer, idx, eight_bit_mode):
    weights = dlayer[0]['kernel_load_cfg']['para_start_addr']
    weights_data = ', '.join([
        ('\n' if i % 64 == 0 else '') +
        signed_to_hex(item, 8 if eight_bit_mode else 16)
        for item, i in zip(weights, range(len(weights)))
    ])
    para_type = 'uint8_t' if eight_bit_mode else 'uint16_t'
    return para_type + \
           ' para_start_addr_{idx}[] __attribute__((aligned(128))) = {{{data}}};'.format(idx=idx, data=weights_data)


def gen_layer_list_code(klayers: [layer_list_to_k210_layer.K210Layer], eight_bit_mode):
    structs = gen_layer_list_struct(klayers)
    output_scale, output_bias = structs[-1][1]

    header_part = '#include "kpu.h"'
    footer_part = '\n'.join([
        'kpu_task_t* kpu_task_init(kpu_task_t* task){',
        ' \n'.join([
            ' la[{idx}].kernel_pool_type_cfg.data.bwsx_base_addr = (uint64_t)&bwsx_base_addr_{idx};\n'
            ' la[{idx}].kernel_calc_type_cfg.data.active_addr = (uint64_t)&active_addr_{idx};\n'
            ' la[{idx}].kernel_load_cfg.data.para_start_addr = (uint64_t)&para_start_addr_{idx};'
                .format(idx=idx)
            for idx in range(len(structs))
        ]),
        ' task->layers = la;',
        ' task->layers_length = sizeof(la)/sizeof(la[0]);',
        ' task->eight_bit_mode = {};'.format(str(1 if eight_bit_mode else 0)),
        ' task->output_scale = {};'.format(output_scale),
        ' task->output_bias = {};'.format(output_bias),
        ' return task;',
        '}'
    ])

    layer_part = 'kpu_layer_argument_t la[] __attribute__((aligned(128))) = {\n' + ',\n'.join([
        gen_layer_code(layer, idx)
        for layer, idx in zip(structs, range(len(structs)))
    ]) + '};'

    bn_part = [
        gen_bn_code(layer, idx)
        for layer, idx in zip(structs, range(len(structs)))
    ]

    act_part = [
        gen_act_code(layer, idx)
        for layer, idx in zip(structs, range(len(structs)))
    ]

    weights_part = [
        gen_weights_code(layer, idx, eight_bit_mode)
        for layer, idx in zip(structs, range(len(structs)))
    ]

    return '\n\n'.join([
        header_part,
        '\n'.join(act_part),
        '\n'.join(bn_part),
        '\n'.join(weights_part),
        layer_part,
        footer_part,
    ])
