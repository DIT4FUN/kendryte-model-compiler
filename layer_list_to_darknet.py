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

import tensor_list_to_layer_list
import numpy


def gen_config_file(layers):
    ret = []
    for layer in layers:
        assert (isinstance(layer, tensor_list_to_layer_list.LayerBase))
        ret.append('[' + layer.name + ']')
        for k, v in layer.config.items():
            ret.append(str(k) + '=' + str(v))
        ret.append('')

    return '\n'.join(ret)


def gen_weights(layers):
    ret = [numpy.array([0, 2, 0, 0], 'int32').tobytes()]  # header

    for layer in layers:
        assert (isinstance(layer, tensor_list_to_layer_list.LayerBase))
        if type(layer) in (
                tensor_list_to_layer_list.LayerNet,
                tensor_list_to_layer_list.LayerMaxpool
        ):
            pass
        elif isinstance(layer, tensor_list_to_layer_list.LayerConvolutional) or \
                isinstance(layer, tensor_list_to_layer_list.LayerDepthwiseConvolutional):
            if str(layer.config['batch_normalize']) != '0':
                gamma = numpy.array(layer.batch_normalize_gamma, 'float32')
                beta = numpy.array(layer.batch_normalize_beta, 'float32')
                bias = numpy.array(layer.batch_normalize_moving_mean, 'float32')
                if layer.bias is not None:
                    bias = bias - numpy.array(layer.bias, 'float32')
                variance = numpy.array(layer.batch_normalize_moving_variance, 'float32')

                ret.append(beta.tobytes())
                ret.append(gamma.tobytes())
                ret.append(bias.tobytes())
                ret.append(variance.tobytes())
            else:
                bias = numpy.array(layer.bias, 'float32')
                ret.append(bias.tobytes())

            weights = numpy.array(layer.weights, 'float32')
            weights_trans = numpy.transpose(weights, [3, 2, 0, 1])
            ret.append(weights_trans.tobytes())
        else:
            print('unknown layer:', layer.name, type(layer))

    return b''.join(ret)
