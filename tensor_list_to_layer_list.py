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
        self.tensor = info
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
        self.tensor = info
        self.bias = None
        batch_norm = None
        activation = None
        bias_add = None
        bn_add, bn_sub, bn_div, bn_mul = None, None, None, None
        leaky_reul_max = None

        if self.type_match(info, ['Add', 'Conv2D']):
            bias_add, conv2d = info
        elif self.type_match(info, ['BiasAdd', 'Conv2D']):
            bias_add, conv2d = info
        elif self.type_match(info, ['Relu', 'BiasAdd', 'Conv2D']):
            activation, bias_add, conv2d = info
        elif self.type_match(info, ['LeakyRelu', 'BiasAdd', 'Conv2D']):
            activation, bias_add, conv2d = info
        elif self.type_match(info, ['Maximum', 'Mul', 'BiasAdd', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, bias_add, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'Add', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, bias_add, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Relu6', 'BiasAdd', 'Conv2D']):
            activation, bias_add, conv2d = info
        elif self.type_match(info, ['Relu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, bias_add, conv2d = info
        elif self.type_match(info, ['Maximum', 'Mul', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, batch_norm, bias_add, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'FusedBatchNorm', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, batch_norm, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
        elif self.type_match(info, ['Maximum', 'Mul', 'Add', 'Mul', 'RealDiv', 'Sub', 'Conv2D']):
            leaky_reul_max, leaky_reul_mul, bn_add, bn_mul, bn_div, bn_sub, conv2d = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
            batch_norm = [bn_add, bn_mul, bn_div, bn_sub]
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            activation, batch_norm, bias_add, conv2d = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'Conv2D']):
            activation, batch_norm, conv2d = info
        else:
            print('not supported convolutional info.')
            return

        self.config['batch_normalize'] = 1 if batch_norm is not None else 0

        self.tensor_conv_w = conv2d.op.inputs[1]
        self.tensor_conv_x = conv2d.op.inputs[0]
        self.tensor_conv_y = conv2d
        # self.tensor_bn = bn_add if bn_add is not None else batch_norm
        if activation is not None:
            self.tensor_activation = activation
        elif batch_norm is not None:
            self.tensor_activation = batch_norm
        elif bias_add is not None:
            self.tensor_activation = bias_add

        assert (isinstance(conv2d, tf.Tensor))
        self.config['size'] = int(conv2d.op.inputs[1].shape[0])
        self.config['stride'] = conv2d.op.get_attr('strides')[1]
        self.config['pad'] = 1 if conv2d.op.get_attr('padding') != 'SAME' else 0
        self.config['filters'] = int(conv2d.shape[3])

        if isinstance(activation, list):
            self.config['activation'] = activation[0]
            self.tensor_activation = activation[1]
        elif activation is not None:
            self.config['activation'] = activation.op.type
        else:
            self.config['activation'] = 'linear'

        self.weights = sess.run(conv2d.op.inputs[1])
        if bias_add is not None:
            self.bias = sess.run(bias_add.op.inputs[1])

        if isinstance(batch_norm, list):
            self.batch_normalize_moving_mean = sess.run(bn_sub.op.inputs[1])
            self.batch_normalize_moving_variance = sess.run(bn_div.op.inputs[1])
            self.batch_normalize_gamma = sess.run(bn_mul.op.inputs[1])
            self.batch_normalize_beta = sess.run(bn_add.op.inputs[1])
        elif batch_norm is not None:
            if 'gamma/read' not in batch_norm.op.inputs[1].name:
                print('[warning] gamma/read should in name:', batch_norm.op.inputs[1].name)
            if 'beta/read' not in batch_norm.op.inputs[2].name:
                print('[warning] beta/read should in name:', batch_norm.op.inputs[1].name)

            self.batch_normalize_gamma = sess.run(batch_norm.op.inputs[1])
            self.batch_normalize_beta = sess.run(batch_norm.op.inputs[2])
            if len(batch_norm.op.inputs) == 5:
                if 'gamma/read' not in batch_norm.op.inputs[1].name:
                    print('[warning] moving_mean/read should in name:', batch_norm.op.inputs[3].name)
                if 'beta/read' not in batch_norm.op.inputs[2].name:
                    print('[warning] moving_variance/read should in name:', batch_norm.op.inputs[4].name)
                self.batch_normalize_moving_mean = sess.run(batch_norm.op.inputs[3])
                self.batch_normalize_moving_variance = sess.run(batch_norm.op.inputs[4])
            else:
                batch_norm_1 = batch_norm.op.outputs[1]
                batch_norm_2 = batch_norm.op.outputs[2]
                batch_normal_outputs = [
                    op for k, op in sess.graph._nodes_by_name.items()
                    if len(op.inputs) == 2 and op.inputs[1] in (batch_norm_1, batch_norm_2)
                ]
                mean_tensor = batch_normal_outputs[0].inputs[0]
                variance_tensor = batch_normal_outputs[1].inputs[0]
                assert ('moving_mean/read' in mean_tensor.name)
                assert ('moving_variance/read' in variance_tensor.name)
                self.batch_normalize_moving_mean = sess.run(mean_tensor)
                self.batch_normalize_moving_variance = sess.run(variance_tensor)

            if self.batch_normalize_moving_mean.size == 0:
                self.batch_normalize_moving_mean = 0
            if self.batch_normalize_moving_variance.size == 0:
                self.batch_normalize_moving_variance = 1

class LayerDepthwiseConvolutional(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.name = 'depthwise_convolutional'
        self.config = {}
        self.tensor = info
        bias_add = None
        if self.type_match(info, ['Relu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bias_add, dwconv = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bias_add, dwconv = info
        elif self.type_match(info, ['Relu6', 'FusedBatchNorm', 'DepthwiseConv2dNative']):
            activation, batch_norm, dwconv = info
        elif self.type_match(info, ['LeakyRelu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            activation, batch_norm, bias_add, dwconv = info
        elif self.type_match(info, ['Maximum', 'Mul', 'Add', 'Mul', 'RealDiv', 'Sub', 'DepthwiseConv2dNative']):
            leaky_reul_max, leaky_reul_mul, bn_add, bn_mul, bn_div, bn_sub, dwconv = info
            activation = ['leaky', leaky_reul_max, leaky_reul_mul]
            batch_norm = [bn_add, bn_mul, bn_div, bn_sub]
        else:
            print('not supported dw_convolutional info.')
            return

        self.config['batch_normalize'] = 1 if batch_norm is not None else 0

        self.tensor_conv_w = dwconv.op.inputs[1]
        self.tensor_conv_x = dwconv.op.inputs[0]
        self.tensor_conv_y = dwconv
        # self.tensor_activation = activation or batch_norm or bais_add
        if activation is not None:
            self.tensor_activation = activation
        elif batch_norm is not None:
            self.tensor_activation = batch_norm
        elif bias_add is not None:
            self.tensor_activation = bias_add

        assert (isinstance(dwconv, tf.Tensor))
        self.config['size'] = int(dwconv.op.inputs[1].shape[0])
        self.config['stride'] = dwconv.op.get_attr('strides')[1]
        self.config['pad'] = 1 if dwconv.op.get_attr('padding') != 'SAME' else 0

        if isinstance(activation, list):
            self.config['activation'] = activation[0]
            self.tensor_activation = activation[1]
        elif activation is not None:
            self.config['activation'] = activation.op.type
        else:
            self.config['activation'] = 'linear'

        self.weights = sess.run(dwconv.op.inputs[1])
        self.bias = sess.run(bias_add.op.inputs[1]) if bias_add else None

        if isinstance(batch_norm, list):
            self.batch_normalize_moving_mean = sess.run(bn_sub.op.inputs[1])
            self.batch_normalize_moving_variance = sess.run(bn_div.op.inputs[1])
            self.batch_normalize_gamma = sess.run(bn_mul.op.inputs[1])
            self.batch_normalize_beta = sess.run(bn_add.op.inputs[1])
        elif batch_norm is not None:
            assert ('gamma/read' in batch_norm.op.inputs[1].name)
            assert ('beta/read' in batch_norm.op.inputs[2].name)
            self.batch_normalize_gamma = sess.run(batch_norm.op.inputs[1])
            self.batch_normalize_beta = sess.run(batch_norm.op.inputs[2])
            if len(batch_norm.op.inputs) == 5:
                assert ('moving_mean/read' in batch_norm.op.inputs[3].name)
                assert ('moving_variance/read' in batch_norm.op.inputs[4].name)
                self.batch_normalize_moving_mean = sess.run(batch_norm.op.inputs[3])
                self.batch_normalize_moving_variance = sess.run(batch_norm.op.inputs[4])
            else:
                batch_norm_1 = batch_norm.op.outputs[1]
                batch_norm_2 = batch_norm.op.outputs[2]
                batch_normal_outputs = [
                    op for k, op in sess.graph._nodes_by_name.items()
                    if len(op.inputs) == 2 and op.inputs[1] in (batch_norm_1, batch_norm_2)
                ]
                mean_tensor = batch_normal_outputs[0].inputs[0]
                variance_tensor = batch_normal_outputs[1].inputs[0]
                assert ('moving_mean/read' in mean_tensor.name)
                assert ('moving_variance/read' in variance_tensor.name)
                self.batch_normalize_moving_mean = sess.run(mean_tensor)
                self.batch_normalize_moving_variance = sess.run(variance_tensor)




class LayerMaxpool(LayerBase):
    def __init__(self, sess, info):
        super().__init__()
        self.name = 'maxpool'
        self.config = {}
        self.tensor = info
        self.tensor_pool = info[0]
        if self.type_match(info, ['MaxPool']):
            max_pool = info[0]
        else:
            print('not supported maxpool info.')
            return

        assert (isinstance(max_pool, tf.Tensor))
        self.config['size'] = max_pool.op.get_attr('ksize')[1]
        self.config['stride'] = max_pool.op.get_attr('strides')[1]


def convert_layer(sess, info):
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


def convert_to_layers(sess, info_list):
    info_list.reverse()
    return [convert_layer(sess, info) for info in info_list]
