from base.base_model import BaseModel
import tensorflow as tf
import tensorflow.contrib.slim as slim
import numpy as np


class TinyYOLOV2(BaseModel):

    MODEL_NAME = 'TinyYOLOv2-Mobilenet'
    object_scale = 5
    no_object_scale = 1
    class_scale = 1
    coordinates_scale = 1

    def __init__(self, config):
        super().__init__(config)
        self.build_model()
        self.init_saver()
        self.net = None

    def build_model(self):
        self.is_training = tf.placeholder(tf.bool)

        self.x = tf.placeholder(tf.float32, shape=[None, self.config.input_width, self.config.input_height, self.config.input_channel] )  # input

        # TODO: Trainner 里加boxes label和cell masks, 不在这里加哦

        # self.y = tf.placeholder(tf.float32, shape=[None, 10])  # output, [batch_size, x, y, w, h, c]
        self.y = tf.placeholder(tf.float32, shape=[None, 13, 13, 5, 5])  # output, [batch_size, x, y, w, h, c]

        def _depthwise_separable_conv(inputs,
                                      num_pwc_filters,
                                      sc,
                                      downsample=False):
            """ Helper function to build the depth-wise separable convolution layer.
            """
            _stride = 2 if downsample else 1

            # skip pointwise by setting num_outputs=None
            depthwise_conv = slim.separable_convolution2d(inputs,
                                                          num_outputs=None,
                                                          stride=_stride,
                                                          depth_multiplier=1,
                                                          kernel_size=[3, 3],
                                                          scope=sc + '/depthwise_conv')

            bn = slim.batch_norm(depthwise_conv, scope=sc + '/dw_batch_norm')

            # TODO: test on relu6 or back to lrelu
            act = tf.nn.relu6(bn)

            pointwise_conv = slim.convolution2d(act,
                                                num_pwc_filters,
                                                kernel_size=[1, 1],
                                                scope=sc + '/pointwise_conv')
            bn = slim.batch_norm(pointwise_conv, scope=sc + '/pw_batch_norm')

            act = tf.nn.relu6(bn)

            return act

        # network architecture
        with tf.variable_scope(self.MODEL_NAME) as sc:
            end_points_collection = sc.name + '_end_points'

            with slim.arg_scope([slim.batch_norm],
                                is_training=self.is_training,
                                activation_fn=tf.nn.leaky_relu,
                                fused=True):
                # 普通卷积
                net = slim.convolution2d(inputs=self.x,
                                         num_outputs=16,
                                         kernel_size=(3,3),
                                         stride=1,
                                         scope='conv_1')

                net = slim.batch_norm(net, scope='conv_1/batchnorm')
                net = slim.max_pool2d(net,
                                      kernel_size=2,
                                      stride=2,
                                      scope='conv_1/maxpool')

                # depthwise separable 卷积
                net = _depthwise_separable_conv(net, 32, sc='conv_ds_2')
                net = slim.max_pool2d(net,
                                      kernel_size=2,
                                      stride=2,
                                      scope='conv_ds_2/maxpool')

                net = _depthwise_separable_conv(net, 64, sc='conv_ds_3')
                net = slim.max_pool2d(net,
                                      kernel_size=2,
                                      stride=2,
                                      scope='conv_ds_3/maxpool')

                net = _depthwise_separable_conv(net, 128, sc='conv_ds_4')
                net = slim.max_pool2d(net,
                                      kernel_size=2,
                                      stride=2,
                                      scope='conv_ds_4/maxpool')

                net = _depthwise_separable_conv(net, 256, sc='conv_ds_5')
                net = slim.max_pool2d(net,
                                      kernel_size=2,
                                      stride=2,
                                      scope='conv_ds_5/maxpool')

                net = _depthwise_separable_conv(net, 512, sc='conv_ds_6')
                net = slim.max_pool2d(net,
                                      kernel_size=2,
                                      stride=1,
                                      scope='conv_ds_6/maxpool')

                net = _depthwise_separable_conv(net, 1024, sc='conv_ds_7')
                net = _depthwise_separable_conv(net, 1024, sc='conv_ds_8')

                # classifier
                #

                net = slim.convolution2d(inputs=net,
                                         num_outputs=20,  # ANCHORS * (CLASSES + 5)
                                         kernel_size=(1,1),
                                         stride=1,
                                         scope='conv_9')

                # net = slim.fully_connected(inputs=net,
                #                      num_outputs=100,
                #                      scope='fc_9')


                # net = tf.reshape(net, shape=(self.config.batch_size, 13, 13, 5, 4))
                net = tf.reshape(net, [self.config.batch_size, -1])
                net = tf.reshape(net, shape=(self.config.batch_size, 13, 13, 5, 4))

                import ipdb; ipdb.set_trace()
                # net = slim.convolution(inputs=self.x,
                #                          num_outputs=100,
                #                          kernel_size=(1, 1),
                #                          stride=1,
                #                          activation_fn=slim.linear,
                #                          scope='conv_9')

                self.net = net


        self.build_train()

    def build_train(self):

        # assert self.net  # make sure build_model was invoked before.

        # define loss func and optimizer
        with tf.name_scope("loss"):
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=self.y, logits=self.net))

            # TODO: auto modulate learning rate.
            # select optimizer
            if self.config.optimizer == 'adam':
                self.train_step = tf.train.AdamOptimizer(self.config.learning_rate).minimize(self.loss,
                                                                                             global_step=self.global_step_tensor)
            elif self.config.optimizer == 'momentum':
                self.train_step = tf.train.MomentumOptimizer(learning_rate=self.config.learning_rate,
                                                             momentum=0.9,
                                                             )

            correct_prediction = tf.equal(tf.argmax(self.net, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def loss_function(self,
             y_true,
             y_pred):

        NORM_H, NORM_W = 416, 416
        ANCHORS = 3
        BOX = 5
        GRID_H, GRID_W = 13, 13
        SCALE_NOOB, SCALE_CONF, SCALE_COOR, SCALE_PROB = 1, 5.0, 5.0, 1.0 # Loss各项权重

        ### Adjust prediction
        # adjust x and y
        # x, y, w, h, c
        # bx, by
        pred_box_xy = tf.sigmoid(y_pred[:, :, :, :, :2])

        '''
        region 层计算
        (tx, ty, tw, th, to)
        bw = p^w*e^(tw)'''
        # adjust w and h
        # bw, bh
        pred_box_wh = tf.exp(y_pred[:, :, :, :, 2:4]) * np.reshape(ANCHORS, [1, 1, 1, BOX, 2])
        pred_box_wh = tf.sqrt(pred_box_wh / np.reshape([float(GRID_W), float(GRID_H)], [1, 1, 1, 1, 2]))

        # bx, by, bw, bh
        y_pred = tf.concat([pred_box_xy, pred_box_wh], 2)

        print("Y_pred shape: {}".format(y_pred.shape))

        ### Adjust ground truth
        # adjust x and y
        center_xy = .5 * (y_true[:, :, :, :, 0:2] + y_true[:, :, :, :, 2:4])
        center_xy = center_xy / np.reshape([(float(NORM_W) / GRID_W), (float(NORM_H) / GRID_H)], [1, 1, 1, 1, 2])
        true_box_xy = center_xy - tf.floor(center_xy)

        # adjust w and h
        true_box_wh = (y_true[:, :, :, :, 2:4] - y_true[:, :, :, :, 0:2])
        true_box_wh = tf.sqrt(true_box_wh / np.reshape([float(NORM_W), float(NORM_H)], [1, 1, 1, 1, 2]))

        # adjust confidence
        pred_tem_wh = tf.pow(pred_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
        pred_box_area = pred_tem_wh[:, :, :, :, 0] * pred_tem_wh[:, :, :, :, 1]
        pred_box_ul = pred_box_xy - 0.5 * pred_tem_wh
        pred_box_bd = pred_box_xy + 0.5 * pred_tem_wh

        true_tem_wh = tf.pow(true_box_wh, 2) * np.reshape([GRID_W, GRID_H], [1, 1, 1, 1, 2])
        true_box_area = true_tem_wh[:, :, :, :, 0] * true_tem_wh[:, :, :, :, 1]
        true_box_ul = true_box_xy - 0.5 * true_tem_wh
        true_box_bd = true_box_xy + 0.5 * true_tem_wh

        intersect_ul = tf.maximum(pred_box_ul, true_box_ul)
        intersect_br = tf.minimum(pred_box_bd, true_box_bd)
        intersect_wh = intersect_br - intersect_ul
        intersect_wh = tf.maximum(intersect_wh, 0.0)
        intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

        iou = tf.truediv(intersect_area, true_box_area + pred_box_area - intersect_area)
        best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
        best_box = tf.to_float(best_box)


        true_box_conf = best_box
        # true_box_conf = tf.expand_dims(best_box * y_true[:, :, :, :, 4], -1)

        # adjust confidence
        true_box_prob = y_true[:, :, :, :, 5:]

        y_true = tf.concat([true_box_xy, true_box_wh, true_box_conf, true_box_prob], 4)
        print("Y_true shape: {}".format(y_true.shape))
        # y_true = tf.Print(y_true, [true_box_wh], message='DEBUG', summarize=30000)

        ### Compute the weights
        weight_coor = tf.concat(4 * [true_box_conf], 4)
        weight_coor = SCALE_COOR * weight_coor

        weight_conf = SCALE_NOOB * (1. - true_box_conf) + SCALE_CONF * true_box_conf

        weight = tf.concat([weight_coor, weight_conf], 4)
        print("Weight shape: {}".format(weight.shape))

        ### Finalize the loss
        loss = tf.pow(y_pred - y_true, 2)
        loss = loss * weight
        loss = tf.reshape(loss, [-1, GRID_W * GRID_H * BOX * (4 + 1)])
        loss = tf.reduce_sum(loss, 1)
        loss = .5 * tf.reduce_mean(loss)

        return loss

    def yolo_loss(self,
                  y_true,
                  y_pred):
        """
        y_true:
            []
        y_pred:
            [x, y, w, h, conf]
        """

        mask_shape = tf.shape(y_true)[:4]

        # grid_w, grid_h = 7, 7 (ref YOLOv1)
        cell_x = tf.to_float(
            tf.reshape(tf.tile(tf.range(self.grid_w), [self.grid_h]), (1, self.grid_h, self.grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))

        # nb_box = len(anchors) // 2
        cell_grid = tf.tile(tf.concat([cell_x, cell_y], -1), [self.batch_size, 1, 1, self.nb_box, 1])


        # 指示函数的值
        coord_mask = tf.zeros(mask_shape)
        conf_mask = tf.zeros(mask_shape)
        class_mask = tf.zeros(mask_shape)

        seen = tf.Variable(0.)
        total_recall = tf.Variable(0.)

        """
        预测结果 region 计算
        """
        ### adjust x and y
        pred_box_xy = tf.sigmoid(y_pred[..., :2]) + cell_grid

        ### adjust w and h
        pred_box_wh = tf.exp(y_pred[..., 2:4]) * np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2])

        ### adjust confidence
        pred_box_conf = tf.sigmoid(y_pred[..., 4])

        ### adjust class probabilities
        pred_box_class = y_pred[..., 5:]

        """
        Adjust ground truth
        """
        ### adjust x and y
        true_box_xy = y_true[..., 0:2]  # relative position to the containing cell

        ### adjust w and h
        true_box_wh = y_true[..., 2:4]  # number of cells accross, horizontally and vertically

        ### adjust confidence
        # 计算IoU
        true_wh_half = true_box_wh / 2.
        true_mins = true_box_xy - true_wh_half
        true_maxes = true_box_xy + true_wh_half

        pred_wh_half = pred_box_wh / 2.
        pred_mins = pred_box_xy - pred_wh_half
        pred_maxes = pred_box_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_box_wh[..., 0] * true_box_wh[..., 1]  # w*h
        pred_areas = pred_box_wh[..., 0] * pred_box_wh[..., 1]  # w*h

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        true_box_conf = iou_scores * y_true[..., 4]

        ### adjust class probabilities
        true_box_class = tf.argmax(y_true[..., 5:], -1)

        """
        确定示数
        """
        ### coordinate mask: simply the position of the ground truth boxes (the predictors)
        coord_mask = tf.expand_dims(y_true[..., 4], axis=-1) * self.coord_scale

        ### confidence mask: penelize predictors + penalize boxes with low IOU
        # penalize the confidence of the boxes, which have IOU with some ground truth box < 0.6
        true_xy = self.true_boxes[..., 0:2]
        true_wh = self.true_boxes[..., 2:4]

        true_wh_half = true_wh / 2.
        true_mins = true_xy - true_wh_half
        true_maxes = true_xy + true_wh_half

        pred_xy = tf.expand_dims(pred_box_xy, 4)
        pred_wh = tf.expand_dims(pred_box_wh, 4)

        pred_wh_half = pred_wh / 2.
        pred_mins = pred_xy - pred_wh_half
        pred_maxes = pred_xy + pred_wh_half

        intersect_mins = tf.maximum(pred_mins, true_mins)
        intersect_maxes = tf.minimum(pred_maxes, true_maxes)
        intersect_wh = tf.maximum(intersect_maxes - intersect_mins, 0.)
        intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

        true_areas = true_wh[..., 0] * true_wh[..., 1]
        pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

        union_areas = pred_areas + true_areas - intersect_areas
        iou_scores = tf.truediv(intersect_areas, union_areas)

        best_ious = tf.reduce_max(iou_scores, axis=4)
        conf_mask = conf_mask + tf.to_float(best_ious < 0.6) * (1 - y_true[..., 4]) * self.no_object_scale

        # penalize the confidence of the boxes, which are reponsible for corresponding ground truth box
        conf_mask = conf_mask + y_true[..., 4] * self.object_scale

        """
        Warm-up training
        """
        no_boxes_mask = tf.to_float(coord_mask < self.coord_scale / 2.)
        seen = tf.assign_add(seen, 1.)

        true_box_xy, true_box_wh, coord_mask = tf.cond(tf.less(seen, self.warmup_batches + 1),
                                                       lambda: [true_box_xy + (0.5 + cell_grid) * no_boxes_mask,
                                                                true_box_wh + tf.ones_like(true_box_wh) * \
                                                                np.reshape(self.anchors, [1, 1, 1, self.nb_box, 2]) * \
                                                                no_boxes_mask,
                                                                tf.ones_like(coord_mask)],
                                                       lambda: [true_box_xy,
                                                                true_box_wh,
                                                                coord_mask])

        """
        Finalize the loss
        """
        nb_coord_box = tf.reduce_sum(tf.to_float(coord_mask > 0.0))
        nb_conf_box = tf.reduce_sum(tf.to_float(conf_mask > 0.0))

        loss_xy = tf.reduce_sum(tf.square(true_box_xy - pred_box_xy) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_wh = tf.reduce_sum(tf.square(true_box_wh - pred_box_wh) * coord_mask) / (nb_coord_box + 1e-6) / 2.
        loss_conf = tf.reduce_sum(tf.square(true_box_conf - pred_box_conf) * conf_mask) / (nb_conf_box + 1e-6) / 2.
        # TODO: 原作者这里是SE, 搞不懂为什么要用cross entropy，量纲应该不一样
        # loss_class = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class)

        loss = tf.cond(tf.less(seen, self.warmup_batches + 1),
                       lambda: loss_xy + loss_wh + loss_conf + 10,
                       lambda: loss_xy + loss_wh + loss_conf)

        return loss


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

