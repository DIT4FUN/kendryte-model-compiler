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


class PbConverter:
    def __init__(self, output_tensor, input_tensor=None):
        self.output_tensor = output_tensor
        self.input_tensor = input_tensor
        self.dst = []

    def ty_match(self, path):
        if self.output_tensor is None:
            return False

        p = self.output_tensor
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
        return self.output_tensor.op.inputs[idx]

    def pop_src(self, *index_list):
        ret = []
        for idx in index_list:
            ret.append(self.output_tensor)
            self.output_tensor = self.get_input(idx)

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
        elif self.ty_match(['Maximum', ('Mul', 1), 'FusedBatchNorm', 'Conv2D']):
            self.dst.append(['convolutional', *self.pop_src(0, 1, 0, 0)])
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
            net = self.output_tensor
            self.dst.append(['net', net])
            self.output_tensor = None
            return True
        else:
            return False

    def try_input(self):
        if self.output_tensor is not None and self.output_tensor == self.input_tensor:
            net = self.output_tensor
            self.dst.append(['net', net])
            self.output_tensor = None
            return True
        else:
            return False

    def convert_step(self):
        converters = (
            self.try_input,
            self.try_reshape,
            self.try_convolutional,
            self.try_maxpool,
            self.try_depthwise_convolutional,
            self.try_placeholder,
        )

        for converter in converters:
            if converter():
                return True

        if self.output_tensor is not None:
            print('no converter for', self.output_tensor.op.type, 'name:', self.output_tensor.op.name)
        else:
            print('convert done.')
        return False

    def convert(self):
        while self.convert_step():
            pass
