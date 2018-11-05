import tensorflow as tf


class PbConverter:
    def __init__(self, target: tf.Tensor):
        self.src = target
        self.dst = []

    def ty_match(self, path):
        if self.src is None:
            return False

        p = self.src
        path[-1] = (path[-1], None) if isinstance(path[-1], str) else (path[-1][0], None)

        for item in path:
            if isinstance(item, str):
                ty, next_input = (item, 0)
            else:
                ty, next_input = item

            if p.op.type != ty:
                return False
            if next_input is not None:
                p = p.op.inputs[next_input]

        return True

    def get_input(self, idx=0):
        return self.src.op.inputs[idx]

    def pop_src(self, *index_list):
        ret = []
        for idx in index_list:
            ret.append(self.src)
            self.src = self.get_input(idx)

        return ret

    def try_reshape(self):
        if self.ty_match(['Reshape']):
            self.pop_src(0)
            # self.dst.append(['???', reshape])
            return True
        else:
            return False

    def try_convolutional(self):
        if self.ty_match(['BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 0)])
            return True
        elif self.ty_match(['Add', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 0)])
            return True
        elif self.ty_match(['Relu', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 0, 0)])
            return True
        elif self.ty_match(['Maximum', ['Mul', 1], 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 1, 0, 0)])
            return True
        elif self.ty_match(['Maximum', ['Mul', 1], 'Add', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 1, 0, 0)])
            return True
        elif self.ty_match(['Relu6', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 0, 0)])
            return True
        elif self.ty_match(['Relu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 0, 0, 0)])
            return True
        elif self.ty_match(['Relu', 'FusedBatchNorm', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 0, 0)])
            return True
        elif self.ty_match(['Maximum', ('Mul', 1), 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 1, 0, 0, 0)])
            return True
        elif self.ty_match(['Maximum', ('Mul', 1), 'Add', 'Mul', 'RealDiv', 'Sub', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 1, 0, 0, 0, 0, 0)])
            return True
        elif self.ty_match(['Relu6', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 0, 0, 0)])
            return True
        elif self.ty_match(['Relu6', 'FusedBatchNorm', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 0, 0)])
            return True
        else:
            return False

    def try_maxpool(self):
        if self.ty_match(['MaxPool']):
            self.dst.append(['maxpool', *self.pop_src(0)])
            return True
        else:
            return False

    def try_depthwise_convolutional(self):
        if self.ty_match(['Relu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            self.dst.append(['depthwise_convolutional', *self.pop_src(0, 0, 0, 0)])
            return True
        elif self.ty_match(['Relu', 'FusedBatchNorm', 'DepthwiseConv2dNative']):
            self.dst.append(['depthwise_convolutional', *self.pop_src(0, 0, 0)])
            return True
        elif self.ty_match(['Relu6', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            self.dst.append(['depthwise_convolutional', *self.pop_src(0, 0, 0, 0)])
            return True
        elif self.ty_match(['Relu6', 'FusedBatchNorm', 'DepthwiseConv2dNative']):
            self.dst.append(['depthwise_convolutional', *self.pop_src(0, 0, 0)])
            return True
        elif self.ty_match(['Maximum', ('Mul', 1), 'Add', 'Mul', 'RealDiv', 'Sub', 'DepthwiseConv2dNative']):
            self.dst.append(['depthwise_convolutional', *self.pop_src(0, 1, 0, 0, 0, 0, 0)])
            return True
        else:
            return False

    def try_placeholder(self):
        if self.ty_match(['Placeholder']):
            net = self.src
            self.dst.append(['net', net])
            self.src = None
            return True
        else:
            return False

    def convert_step(self):
        converters = (
            self.try_reshape,
            self.try_convolutional,
            self.try_maxpool,
            self.try_depthwise_convolutional,
            self.try_placeholder,
        )

        for converter in converters:
            if converter():
                return True

        if self.src is not None:
            print('no converter for', self.src.op.type, 'name:', self.src.op.name)
        else:
            print('convert done.')
        return False

    def convert(self):
        while self.convert_step():
            pass
