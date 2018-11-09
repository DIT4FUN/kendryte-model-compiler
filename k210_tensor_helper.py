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

def k210_sub_layer_conv(prev, weights, strides):
    return tf.nn.conv2d(
        prev, weights,
        strides=[1, strides, strides, 1],
        padding='SAME'
    )


def k210_sub_layer_bn(prev, mean, variance, offset, scale, epsilon=1e-6):
    return tf.nn.batch_normalization(prev, mean, variance, offset, scale, epsilon)


def k210_layer(prev, conv_args, bn_args, activation_function, pooling_function):
    if isinstance(activation_function, str):
        activation_function = {
            'linear': (lambda x:x),
            'relu': tf.nn.relu,
            'relu6': tf.nn.relu6,
            'leaky_relu': tf.nn.leaky_relu,
        }[activation_function]

    if isinstance(pooling_function, str):
        pooling_function = {
            'maxpool': tf.nn.max_pool
        }[pooling_function]

    l1 = k210_sub_layer_conv(prev, **conv_args)
    l2 = k210_sub_layer_bn(l1, **bn_args)
    l3 = activation_function(l2)
    l4 = pooling_function(l3)
    return l4
