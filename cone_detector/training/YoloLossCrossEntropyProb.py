import logging

import numpy as np
import tensorflow as tf

log = logging.getLogger()


# TODO fai bese class per la loss!!
class YoloLossCrossEntropyProb(object):
    def __init__(self, parameters):
        self.parameters = parameters
        # TODO: check that these true values are eactually what the loss expect to receive and that they work good

    def set_loss_placeholder(self):
        with tf.device("/gpu:0"):
            log.info("Creating loss true values placeholder...")
            self.true_values_ph = tf.placeholder(shape=self.parameters.true_values_shape, dtype=tf.float32, name='label_palceholder')

    def get_loss(self, net_output):
        self.set_loss_placeholder()
        loss_output = self.build_loss(net_output, self.true_values_ph)

        # log.info must be on the cpu
        with tf.device("/cpu:0"):
            loss_output = tf.Print(loss_output, [loss_output], message='Training loss: ')

        return loss_output, self.true_values_ph

    def build_loss(self, net_output, true_values):
        with tf.name_scope("Loss"):

            output_w = float(self.parameters.output_w)
            output_h = float(self.parameters.output_h)
            n_classes = self.parameters.n_classes
            use_sqrt_loss = self.parameters.use_sqrt_loss
            n_anchors = self.parameters.n_anchors
            rescaled_anchors = np.reshape(self.parameters.anchors, [1, 1, 1, n_anchors, 2]) / np.reshape([output_w, output_h], [1, 1, 1, 1, 2])
            exp_cap = 50
            xy_range_start = 5 + n_classes
            xy_range_end = 5 + n_classes + 2

            # Adjust prediction
            # adjust x and y
            pred_box_xy_rel = tf.sigmoid(net_output[:, :, :, :, :2])

            # changed from original loss - network prediction directly in dimension one
            pred_box_wh_oneb = tf.exp(tf.minimum(net_output[:, :, :, :, 2:4], exp_cap)) * rescaled_anchors

            if use_sqrt_loss is False:
                log.info("Using loss without sqrt")
                pred_box_wh_oneb_sqrt = pred_box_wh_oneb
            elif use_sqrt_loss is True:
                log.info("Using loss with sqrt")
                pred_box_wh_oneb_sqrt = tf.sqrt(pred_box_wh_oneb)
            else:
                raise ValueError("Invalid use_sqrt_loss value!")

            # adjust confidence
            pred_box_conf = tf.expand_dims(tf.sigmoid(net_output[:, :, :, :, 4]), -1)

            # adjust probability
            # pred_box_prob = tf.nn.softmax(net_output[:, :, :, :, 5:5 + n_classes])
            pred_box_prob = net_output[:, :, :, :, 5:5 + n_classes]

            # Network ouput loss term completed
            pred_loss = tf.concat([pred_box_xy_rel, pred_box_wh_oneb_sqrt, pred_box_conf], 4)

            # adjust w and h
            # TODO move all this into input placeholder to improve performance
            true_box_wh_oneb = true_values[:, :, :, :, 2:4] - true_values[:, :, :, :, 0:2]

            if use_sqrt_loss is False:
                true_box_wh_oneb_sqrt = true_box_wh_oneb
            elif use_sqrt_loss is True:
                true_box_wh_oneb_sqrt = tf.sqrt(true_box_wh_oneb)
            else:
                raise ValueError("Invalid use_sqrt_loss value!")

            # adjust confidence
            pred_box_wh_grid = pred_box_wh_oneb * np.reshape([output_w, output_h], [1, 1, 1, 1, 2])

            pred_box_area_grid = pred_box_wh_grid[:, :, :, :, 0] * pred_box_wh_grid[:, :, :, :, 1]
            pred_box_ul_grid = pred_box_xy_rel - 0.5 * pred_box_wh_grid
            pred_box_bd_grid = pred_box_xy_rel + 0.5 * pred_box_wh_grid

            # TODO move all this into input placeholder to improve performance
            true_box_xy_rel = true_values[:, :, :, :, xy_range_start:xy_range_end]
            true_box_wh_grid = true_box_wh_oneb * np.reshape([output_w, output_h], [1, 1, 1, 1, 2])
            true_box_area_grid = true_box_wh_grid[:, :, :, :, 0] * true_box_wh_grid[:, :, :, :, 1]
            true_box_ul_rel = true_box_xy_rel - 0.5 * true_box_wh_grid
            true_box_bd_rel = true_box_xy_rel + 0.5 * true_box_wh_grid

            intersect_ul = tf.maximum(pred_box_ul_grid, true_box_ul_rel)
            intersect_br = tf.minimum(pred_box_bd_grid, true_box_bd_rel)
            intersect_wh = intersect_br - intersect_ul
            intersect_wh = tf.maximum(intersect_wh, 0.0)
            intersect_area = intersect_wh[:, :, :, :, 0] * intersect_wh[:, :, :, :, 1]

            iou = tf.truediv(intersect_area, true_box_area_grid + pred_box_area_grid - intersect_area)
            # zero_iou = tf.zeros_like(iou_nan)

            # iou_no_nan = tf.where(tf.is_nan(iou_nan), zero_iou, iou_nan)
            # iou = tf.where(tf.is_inf(iou_no_nan), zero_iou, iou_no_nan)

            best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
            best_box = tf.to_float(best_box)
            true_box_conf = tf.expand_dims(best_box * true_values[:, :, :, :, 4], -1)

            # adjust confidence
            true_box_prob = true_values[:, :, :, :, 5:5 + n_classes]

            true_values_loss = tf.concat([true_box_xy_rel, true_box_wh_oneb_sqrt, true_box_conf], 4)

            # Compute the weights
            weight_coor = tf.concat(4 * [true_box_conf], 4)
            weight_coor = self.parameters.scale_coor * weight_coor
            weight_conf = self.parameters.scale_noob * (1. - true_box_conf) + self.parameters.scale_conf * true_box_conf
            weight_prob = tf.concat(self.parameters.n_classes * [true_box_conf], 4)
            weight_prob = tf.squeeze(self.parameters.scale_proob * true_box_conf)

            weight = tf.concat([weight_coor, weight_conf], 4)

            # Finalize the loss
            loss = tf.pow(pred_loss - true_values_loss, 2)
            loss = loss * weight
            loss = tf.reshape(loss, [-1, int(output_w) * int(output_h) * self.parameters.n_anchors * (4 + 1)])
            loss = tf.reduce_sum(loss, 1)

            sfmce = tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_box_prob[:, :, :, :, :], logits=pred_box_prob[:, :, :, :, :], dim=4)

            loss = .5 * tf.reduce_mean(loss) + tf.reduce_mean(weight_prob * sfmce) + tf.reduce_mean(tf.nn.relu(net_output[:, :, :, :, 2:4] - exp_cap))

            log.info("True_values shape is: {}".format(true_values.shape))
            log.info("Net_output shape: {}".format(net_output.shape))
            log.info("Weight shape: {}".format(weight.shape))

        return loss