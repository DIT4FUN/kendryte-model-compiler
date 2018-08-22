import tensorflow as tf


class GraphConverter:
    def __init__(self, target:tf.Operation):
        self.op = target
        self.dst = []

    def ty_match(self, path):
        if self.op is None:
            return False

        p = self.op
        path[-1] = (path[-1], None) if isinstance(path[-1], str) else (path[-1][0], None)

        for item in path:
            if isinstance(item, str):
                ty, next_input = (item, 0)
            else:
                ty, next_input = item

            if p.type != ty:
                return False
            if next_input is not None:
                p = p.inputs[next_input].op

        return True

    def pop_op(self, *index_list):
        ret = []
        for idx in index_list:
            ret.append(self.op)
            self.op = self.op_input(idx)

        return ret

    def op_input(self, idx=0):
        return self.op.inputs[idx].op

    def try_reshape(self):
        if self.ty_match(['Reshape']):
            self.pop_op(0)
            # self.dst.append(['???', reshape])
            return True
        else:
            return False

    def try_convolutional(self):
        if self.ty_match(['BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_op(0, 0)])
            return True
        elif self.ty_match(['Relu', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_op(0, 0, 0)])
            return True
        elif self.ty_match(['LeakyRelu', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_op(0, 0, 0)])
            return True
        elif self.ty_match(['Relu6', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_op(0, 0, 0)])
            return True
        elif self.ty_match(['Relu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_op(0, 0, 0, 0)])
            return True
        elif self.ty_match(['LeakyRelu', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_op(0, 0, 0, 0)])
            return True
        elif self.ty_match(['Relu6', 'FusedBatchNorm', 'BiasAdd', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_op(0, 0, 0, 0)])
            return True
        else:
            return False

    def try_maxpool(self):
        if self.ty_match(['MaxPool']):
            self.dst.append(['maxpool', *self.pop_op(0)])
            return True
        else:
            return False

    def try_depthwise_convolutional(self):
        if self.ty_match(['Relu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            self.dst.append(['depthwise_convolutional', *self.pop_op(0, 0, 0, 0)])
            return True
        elif self.ty_match(['Relu6', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            self.dst.append(['depthwise_convolutional', *self.pop_op(0, 0, 0, 0)])
            return True
        elif self.ty_match(['LeakyRelu', 'FusedBatchNorm', 'BiasAdd', 'DepthwiseConv2dNative']):
            self.dst.append(['depthwise_convolutional',  *self.pop_op(0, 0, 0, 0)])
            return True
        else:
            return False

    def try_placeholder(self):
        if self.ty_match(['Placeholder']):
            net = self.op
            self.dst.append(['net', net])
            self.op = None
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

        if self.op:
            print('no converter for', self.op.type, 'name:', self.op.name)
        else:
            print('convert done.')
        return False

    def convert_all(self):
        while self.convert_step():
            pass
