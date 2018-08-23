import level2_layers


def gen_config_file(layers):
    ret = []
    for layer in layers:
        assert(isinstance(layer, level2_layers.LayerBase))
        ret.append('[' + layer.name + ']')
        for k, v in layer.config.items():
            ret.append(str(k) + '=' + str(v))
        ret.append('')

    return '\n'.join(ret)

