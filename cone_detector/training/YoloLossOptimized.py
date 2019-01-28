import logging
import tensorflow.contrib.slim as slim
import numpy as np
import tensorflow as tf

log = logging.getLogger()


# TODO fai bese class per la loss!!
class YoloLossOptimized(object):
    def __init__(self, parameters):
        self.parameters = parameters
        # TODO: check that these true values are eactually what the loss expect to receive and that they work good

    def set_loss_placeholder(self):
        with tf.device(self.parameters.tf_device):
            log.info("Creating loss true values placeholder...")
            self.true_values_ph = tf.placeholder(shape=self.parameters.true_values_shape, dtype=tf.float32, name='label_palceholder')

    def get_loss(self, net_output):
        self.set_loss_placeholder()
        loss_output = self.build_loss(net_output, self.true_values_ph)

        return loss_output, self.true_values_ph

    def build_loss(self, net_output, true_values):
        with tf.name_scope("YoloLossOptimized"):

            #with tf.device("/cpu:0"):
            #    net_output = tf.Print(net_output, [tf.shape(net_output) ],  # 0
            #
            #                    message='tf shape: ')


            net_output = tf.cast(net_output, tf.float32)

            output_w = float(self.parameters.output_w)
            output_h = float(self.parameters.output_h)
            n_classes = self.parameters.n_classes
            use_sqrt_loss = self.parameters.use_sqrt_loss
            n_anchors = self.parameters.n_anchors

            reshaped_anchors = np.reshape(np.array(self.parameters.anchors, dtype=np.float32), [1, 1, 1, n_anchors, 2])
            # rescaled_anchors = reshaped_anchors

            exp_cap = 5
            xy_range_start = 5 + n_classes
            xy_range_end = 5 + n_classes + 2

            # Adjust prediction
            # adjust x and y
            pred_box_xy_rel = tf.sigmoid(net_output[:, :, :, :, 0:2])

            # with tf.device("/cpu:0"):
            #    pred_box_xy_rel = tf.Print(pred_box_xy_rel, [tf.reduce_sum(tf.to_float(tf.is_nan(net_output)))],
            #                                      message='nancheck: ')

            # changed from original loss - network prediction directly in dimension one
            exp_cap_net_output = tf.minimum(net_output[:, :, :, :, 2:4], exp_cap)

            pred_box_wh_grid = tf.exp(exp_cap_net_output) * reshaped_anchors
            pred_box_wh_oneb = pred_box_wh_grid / np.reshape(np.array([output_w, output_h], dtype=np.float32), [1, 1, 1, 1, 2])

            if use_sqrt_loss is False:
                log.info("Using loss without sqrt")
                pred_box_wh_oneb_sqrt = pred_box_wh_oneb

            elif use_sqrt_loss is True:
                log.info("Using loss with sqrt")
                pred_box_wh_oneb_sqrt = tf.sqrt(pred_box_wh_oneb)
            else:
                raise ValueError("Invalid use_sqrt_loss value!")

            # adjust confidence
            pred_box_conf = tf.sigmoid(net_output[:, :, :, :, 4])




            # adjust w and h
            # TODO move all this into input placeholder to improve performance
            true_box_wh_oneb = true_values[:, :, :, :, 2:4] - true_values[:, :, :, :, 0:2]
            true_box_wh_grid = (true_values[:, :, :, :, 2:4] - true_values[:, :, :, :, 0:2]) * np.reshape(np.array([output_w, output_h], dtype=np.float32), [1, 1, 1, 1, 2])

            if use_sqrt_loss is False:
                true_box_wh_oneb_sqrt = true_box_wh_oneb
            elif use_sqrt_loss is True:
                true_box_wh_oneb_sqrt = tf.sqrt(true_box_wh_oneb)
            else:
                raise ValueError("Invalid use_sqrt_loss value!")

            # adjust confidence

            pred_box_area_grid = pred_box_wh_grid[:, :, :, :, 0] * pred_box_wh_grid[:, :, :, :, 1]

            pred_box_wh_grid_half = 0.5 * pred_box_wh_grid
            pred_box_mins_grid = pred_box_xy_rel - pred_box_wh_grid_half  # ul
            pred_box_maxs_grid = pred_box_xy_rel + pred_box_wh_grid_half  # br

            # TODO move all this into input placeholder to improve performance
            true_box_xy_rel = true_values[:, :, :, :, xy_range_start:xy_range_end]
            true_box_area_grid = true_box_wh_grid[:, :, :, :, 0] * true_box_wh_grid[:, :, :, :, 1]

            true_box_wh_half_grid = 0.5 * true_box_wh_grid
            true_box_mins_grid = true_box_xy_rel - true_box_wh_half_grid  # ul
            true_box_maxs_grid = true_box_xy_rel + true_box_wh_half_grid  # br

            intersect_mins_grid = tf.maximum(true_box_mins_grid, pred_box_mins_grid)  # ul
            intersect_maxs_grid = tf.minimum(true_box_maxs_grid, pred_box_maxs_grid)  # br

            intersect_wh_grid = intersect_maxs_grid - intersect_mins_grid
            intersect_wh_grid = tf.maximum(intersect_wh_grid, 0.0)
            intersect_area_grid = intersect_wh_grid[:, :, :, :, 0] * intersect_wh_grid[:, :, :, :, 1]

            total_area_grid = true_box_area_grid + pred_box_area_grid - intersect_area_grid

            iou_scores = tf.truediv(intersect_area_grid, total_area_grid)

            max_iou = tf.reduce_max(iou_scores, [3], True)

            best_box = tf.equal(iou_scores, max_iou)

            best_box = tf.to_float(best_box)  # * iou_threshold_mask

            true_box_conf = best_box * true_values[:, :, :, :, 4]  # multiply by confidence - to gate out cells that dont have any box inside


            # adjust confidence
            true_box_prob = true_values[:, :, :, :, 5:5 + n_classes]
            # adjust probability
            pred_box_prob = tf.nn.softmax(net_output[:, :, :, :, 5:5 + n_classes])
            #pred_box_prob = net_output[:, :, :, :, 5:5 + n_classes]
            # Finalize the loss
            # get the individual terms and merge them
            #sfmce = true_box_conf * tf.nn.softmax_cross_entropy_with_logits_v2(labels=true_box_prob, logits=pred_box_prob, dim=4)
            true_box_conf_expanded = tf.expand_dims(true_box_conf, -1)

            probl_loss = tf.concat(n_classes * [true_box_conf_expanded], 4) * tf.square(pred_box_prob - true_box_prob)


            xy_loss_matrix_pre_sum = tf.square(pred_box_xy_rel - true_box_xy_rel)

            xy_loss_matrix = true_box_conf * (xy_loss_matrix_pre_sum[:, :, :, :, 0] + xy_loss_matrix_pre_sum[:, :, :, :, 1])




            wh_loss_matrix_pre_sum = tf.square(pred_box_wh_oneb_sqrt - true_box_wh_oneb_sqrt)
            wh_loss_matrix = true_box_conf * (wh_loss_matrix_pre_sum[:, :, :, :, 0] + wh_loss_matrix_pre_sum[:, :, :, :, 1])

            conf_diff_pow = tf.square(pred_box_conf - true_box_conf)


            conf_loss_obj = true_box_conf * conf_diff_pow

            conf_loss_noobj = (1. - true_box_conf) * conf_diff_pow

            xy_loss = self.parameters.scale_coor * self.flatten_and_reduce_sum(xy_loss_matrix)
            wh_loss = self.parameters.scale_coor * self.flatten_and_reduce_sum(wh_loss_matrix)

            conf_ob_loss = self.parameters.scale_conf * self.flatten_and_reduce_sum(conf_loss_obj)
            conf_noob_loss = self.parameters.scale_noob * self.flatten_and_reduce_sum(conf_loss_noobj)

            sfmce_loss = self.parameters.scale_proob * self.flatten_and_reduce_sum(probl_loss)

            exp_cap_loss = self.flatten_and_reduce_sum(tf.nn.relu(net_output[:, :, :, :, 2:4] - exp_cap))

            yolo_loss = xy_loss + wh_loss + conf_ob_loss + conf_noob_loss + sfmce_loss + exp_cap_loss

            reg_loss = tf.losses.get_regularization_loss()

            loss = .5 * tf.reduce_mean(yolo_loss) + reg_loss



            # log.info must be on the cpu
            # with tf.device("/cpu:0"):
            #             #     loss = tf.Print(loss, [#tf.reduce_mean(loss),
            #             #                            tf.reduce_mean(xy_loss),  # 1.49
            #             #                            tf.reduce_mean(wh_loss),  # 0.45
            #             #                            tf.reduce_mean(conf_ob_loss),  # 31
            #             #                            tf.reduce_mean(conf_noob_loss),  # 7
            #             #                            tf.reduce_mean(sfmce_loss),  # 20
            #             #                            #tf.reduce_mean(exp_cap_loss)
            #             #                            ],  # 0
            #             #
            #             #                     message='Losses: ')

            log.info("Regularizer losses: {}".format(reg_loss))
            log.info("True_values shape is: {}".format(true_values.shape))
            log.info("Net_output shape: {}".format(net_output.shape))

        return loss

    def flatten_and_reduce_sum(self, tensor):
        flattened = tf.layers.flatten(tensor)
        reduced_sum = tf.reduce_sum(flattened, axis=1, keepdims=True)

        return reduced_sum
