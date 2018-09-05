import level2_layers
import numpy


def gen_config_file(layers):
    ret = []
    for layer in layers:
        assert (isinstance(layer, level2_layers.LayerBase))
        ret.append('[' + layer.name + ']')
        for k, v in layer.config.items():
            ret.append(str(k) + '=' + str(v))
        ret.append('')

    return '\n'.join(ret)


def gen_weights(layers):
    ret = [numpy.array([0, 2, 0, 0], 'int32').tobytes()]  # header

    for layer in layers:
        assert (isinstance(layer, level2_layers.LayerBase))
        if type(layer) in (
                level2_layers.LayerNet,
                level2_layers.LayerMaxpool
        ):
            pass
        elif isinstance(layer, level2_layers.LayerConvolutional) or \
                isinstance(layer, level2_layers.LayerDepthwiseConvolutional):
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
