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

        return loss_output, self.true_values_ph

    def build_loss(self, net_output, true_values):
        with tf.name_scope("YoloLossCrossEntropyProb"):
            net_output = tf.cast(net_output, tf.float32)

            output_w = float(self.parameters.output_w)
            output_h = float(self.parameters.output_h)
            n_classes = self.parameters.n_classes
            use_sqrt_loss = self.parameters.use_sqrt_loss
            n_anchors = self.parameters.n_anchors
            rescaled_anchors = np.reshape(self.parameters.anchors, [1, 1, 1, n_anchors, 2]) / np.reshape([output_w, output_h], [1, 1, 1, 1, 2])
            exp_cap = 25
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
            pred_box_conf_sig = tf.sigmoid(net_output[:, :, :, :, 4])
            pred_box_conf = tf.expand_dims(pred_box_conf_sig, -1)

            # adjust probability
            # pred_box_prob = tf.nn.softmax(net_output[:, :, :, :, 5:5 + n_classes])
            pred_box_prob = net_output[:, :, :, :, 5:5 + n_classes]

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

            iou = pred_box_conf_sig * tf.truediv(intersect_area, true_box_area_grid + pred_box_area_grid - intersect_area)

            best_box = tf.equal(iou, tf.reduce_max(iou, [3], True))
            best_box = tf.to_float(best_box)
            best_box_true_values = best_box * true_values[:, :, :, :, 4]  # multiply by confidence
            true_box_conf = tf.expand_dims(best_box_true_values, -1)

            # adjust confidence
            true_box_prob = true_values[:, :, :, :, 5:5 + n_classes]

            # Finalize the loss
            # get the individual terms and merge them
            sfmce = best_box_true_values * tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_box_prob, logits=pred_box_prob, dim=4)

            xy_loss_matrix = true_box_conf * tf.pow(pred_box_xy_rel - true_box_xy_rel, 2)
            wh_loss_matrix = self.parameters.scale_coor * true_box_conf * tf.pow(pred_box_wh_oneb_sqrt - true_box_wh_oneb_sqrt, 2)

            conf_diff_pow = tf.pow(pred_box_conf - true_box_conf, 2)
            conf_diff_pow_times_conf = true_box_conf * conf_diff_pow

            conf_loss_obj = conf_diff_pow_times_conf
            conf_loss_noobj = conf_diff_pow - conf_diff_pow_times_conf  # equivalent to (1. - true_box_conf) * conf_diff_pow

            xy_loss = self.parameters.scale_coor * self.flatten_and_reduce_sum(xy_loss_matrix)
            wh_loss = self.parameters.scale_coor * self.flatten_and_reduce_sum(wh_loss_matrix)

            conf_ob_loss = self.parameters.scale_noob * self.flatten_and_reduce_sum(conf_loss_noobj)
            conf_noob_loss = self.parameters.scale_conf * self.flatten_and_reduce_sum(conf_loss_obj)

            sfmce_loss = self.parameters.scale_proob * self.flatten_and_reduce_sum(sfmce)

            exp_cap_loss = (self.parameters.scale_coor + self.parameters.scale_noob + self.parameters.scale_conf + self.parameters.scale_proob) \
                           * self.flatten_and_reduce_sum(tf.nn.relu(net_output[:, :, :, :, 2:4] - exp_cap))




            yolo_loss = xy_loss + wh_loss + conf_ob_loss + conf_noob_loss + sfmce_loss + exp_cap_loss

            reg_loss = tf.losses.get_regularization_loss()
            log.info("Regularizer loss found: {}".format(reg_loss))
            loss = .5 * tf.reduce_mean(yolo_loss) + reg_loss

            # log.info must be on the cpu
            # with tf.device("/cpu:0"):
            # loss = tf.Print(loss, [loss, xy_loss, wh_loss, conf_ob_loss, conf_noob_loss, sfmce_loss, exp_cap_loss], message='Losses')
            # loss = tf.Print(loss, [loss], message='Loss: ')

            log.info("True_values shape is: {}".format(true_values.shape))
            log.info("Net_output shape: {}".format(net_output.shape))

        return loss

    def flatten_and_reduce_sum(self, tensor):
        shape = tensor.get_shape().as_list()  # a list: [None, 9, 2]
        dim = np.prod(shape[1:])  # dim = prod(9,2) = 18
        reshaped = tf.reshape(tensor, [-1, dim])  # -1 means "all
        return tf.reduce_sum(reshaped, axis=1)
