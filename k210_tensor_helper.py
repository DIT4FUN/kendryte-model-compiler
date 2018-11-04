import tensorflow as tf

def k210_sub_layer_conv(prev, weights, strides, padding):
    return tf.nn.conv2d(
        prev, weights,
        strides=[1, strides, strides, 1],
        padding=padding
    )


def k210_sub_layer_bn(prev, mean, variance, scale, epsilon=1e-6):
    return tf.multiply((prev-mean)/(tf.sqrt(variance)+epsilon), scale)


def k210_sub_layer_pool(prev):
    return prev


def k210_layer(prev, conv_args, bn_args, activation_function, pooling_function):
    l1 = k210_sub_layer_conv(prev, **conv_args)
    l2 = k210_sub_layer_bn(l1, **bn_args)
    l3 = activation_function(l2)
    l4 = pooling_function(l3)
    return l4
