import logging

import numpy as np
import tensorflow as tf

log = logging.getLogger()
from visualization.BoundBox import BoundBox


class Predict(object):

    def __init__(self, parameters, preprocessor, network):
        self.parameters = parameters
        self.preprocessor = preprocessor
        self.network_loaded = False
        self.network = network
        anchors = self.parameters.anchors
        anchors_w = list()
        anchors_h = list()
        for b in range(self.parameters.n_anchors):
            sel_anchor_w = anchors[2 * b + 0]
            anchors_w.append(sel_anchor_w)

            sel_anchor_h = anchors[2 * b + 1]
            anchors_h.append(sel_anchor_h)

        self.anchors_h = np.array(anchors_h, dtype=np.float32)
        self.anchors_w = np.array(anchors_w, dtype=np.float32)


    # @measure_time
    def network_output_pipeline(self, images, pure_cv2_images, train_sess=None):
        # image_to_visualize = self.preprocessor.read_image(image_path=image_path)

        if self.parameters.training is False:
            network_output = self.run_network(images=images, pure_cv2_images=pure_cv2_images)
        else:
            network_output = self.run_net_in_training(images=images, pure_cv2_images=pure_cv2_images,
                                                      train_sess=train_sess)

        output_boxes = self.get_output_boxes(net_output=network_output, images=images)
        output_boxes = self.non_max_suppression(images_boxes=output_boxes)
        # boxes_to_print = self.get_final_boxes(boxes_all_images=output_boxes)
        return output_boxes

    def load_network(self):
        checkpoint = self.parameters.checkpoint
        metagraph = self.parameters.metagraph
        if self.parameters.import_graph_from_metafile is True:

            # import the graph from a meta file
            saver = tf.train.import_meta_graph(metagraph)
            self.sess = tf.Session()
            saver.restore(self.sess, checkpoint)

            graph = tf.get_default_graph()
            self.image = graph.get_tensor_by_name("image_placeholder:0")
            self.train_flag = graph.get_tensor_by_name("flag_placeholder:0")
            self.output_node = graph.get_tensor_by_name('network_output:0')

        elif self.parameters.import_graph_from_metafile is False and self.parameters.weights_from_npy is False:

            # rebuild the graph => if you want e.g. to change the input tensor shape
            self.output_node, self.image, self.train_flag = self.network.get_network()
            self.sess = tf.Session()
            with tf.device("/cpu:0"):
                saver = tf.train.Saver()
                saver.restore(self.sess, checkpoint)

        else:
            self.output_node, self.image, self.train_flag = self.network.get_network()
            self.sess = tf.Session()
            self.sess.run(tf.global_variables_initializer())
        if self.parameters.save_as_graphdef is True:
            tf.train.write_graph(self.sess.graph_def, self.parameters.saved_model_dir, 'model.pbtxt')

        self.network_loaded = True

    # @measure_time
    def run_network(self, images, pure_cv2_images):

        if self.network_loaded is False:
            self.load_network()

        images_to_network_list = list()

        for image, pure_cv2_image in zip(images, pure_cv2_images):
            image_to_network = self.preprocessor.preprocess_for_inference(image=image, pure_cv2_image=pure_cv2_image)
            images_to_network_list.append(image_to_network)

        stacked_images = np.stack(images_to_network_list)

        # image_to_see = cv2.cvtColor(image_to_network, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image',image_to_see)
        # cv2.waitKey()

        network_output = self.sess.run(fetches=self.output_node,
                                       feed_dict={self.image: stacked_images,
                                                  self.train_flag: self.parameters.training})
        # var = [v for v in tf.trainable_variables() if v.name == "tiny_yolo_on_proteins/conv1/weights"]
        # print(self.sess.run(tf.trainable_variables()))
        return network_output

    # @measure_time
    def run_net_in_training(self, images, pure_cv2_images, train_sess):
        graph = tf.get_default_graph()
        image_ph = graph.get_tensor_by_name("image_placeholder:0")
        train_flag_ph = graph.get_tensor_by_name("flag_placeholder:0")
        output_node_ph = graph.get_tensor_by_name('network_output:0')

        images_to_network_list = list()

        for image, pure_cv2_image in zip(images, pure_cv2_images):
            image_to_network = self.preprocessor.preprocess_for_inference(image=image, pure_cv2_image=pure_cv2_image)
            images_to_network_list.append(image_to_network)

        stacked_images = np.stack(images_to_network_list)
        network_output = train_sess.run(fetches=output_node_ph, feed_dict={image_ph: stacked_images, train_flag_ph: False})
        return network_output

    def sigmoid(self, x):
        x = 1. / (1. + np.exp(-x))
        return x

    def softmax(self, x):
        exp_x = np.exp(x)
        x = exp_x / np.sum(exp_x, axis=-1, keepdims=True)
        return x

    # @measure_time

    def get_output_boxes(self, net_output, images):
        n_classes = self.parameters.n_classes
        output_h = self.parameters.output_h
        output_w = self.parameters.output_w
        n_anchors = self.parameters.n_anchors
        boxes_for_all_images = list()

        confidences = self.sigmoid(net_output[:, :, :, :, 4])

        for image_idx in range(len(net_output)):
            boxes = []

            # interpret the output by the network
            for row in range(output_h):
                for col in range(output_w):
                    for b in range(n_anchors):
                        # filter them out if we are not going to use them anyway
                        if confidences[image_idx, row, col, b] >= self.parameters.conf_threshold:
                            probs = self.softmax(net_output[image_idx, row, col, b, 5:5 + n_classes])

                            max_prob = np.amax(probs)
                            class_type = np.argmax(probs)  # get class
                            probs[probs < max_prob] = 0

                            conf = self.sigmoid(net_output[image_idx, row, col, b, 4])

                            assert conf >= self.parameters.conf_threshold

                            p_x, p_y, p_w, p_h = net_output[image_idx, row, col, b, 0:4]

                            image_shape_x = images[image_idx].shape[1]
                            image_shape_y = images[image_idx].shape[0]

                            # The values are in grid space
                            x = col + self.sigmoid(p_x)
                            y = row + self.sigmoid(p_y)
                            w = self.anchors_w[b] * np.exp(p_w)
                            h = self.anchors_h[b] * np.exp(p_h)

                            half_w = w / 2.
                            half_h = h / 2.

                            # The values are in input image space
                            xmin = (x - half_w) * image_shape_x / output_w
                            xmax = (x + half_w) * image_shape_x / output_w
                            ymin = (y - half_h) * image_shape_y / output_h
                            ymax = (y + half_h) * image_shape_y / output_h

                            # log.warn("x {} y {} w {} h {}".format(x, y, w, h))
                            # log.warn("xmin {} xmax {} ymin {} ymax {}".format(xmin, xmax, ymin, ymax))

                            box = BoundBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, probs=probs, class_type=class_type, conf=conf)

                            boxes.append(box)

            boxes_for_all_images.append(boxes)
        return boxes_for_all_images

    def non_max_suppression(self, images_boxes):
        if self.parameters.keep_small_ones is True:
            return self.non_max_suppr_keep_small_ones(images_boxes)
        else:
            return self.non_max_suppr_discard_small_ones(images_boxes)

    def non_max_suppr_keep_small_ones(self, images_boxes):
        n_classes = self.parameters.n_classes
        iou_threshold = self.parameters.iou_threshold
        for image_idx in range(len(images_boxes)):

            for c in range(n_classes):
                # for each class get a list of indices that allow to access the images_boxes[image_idx] in the
                # order of probability per class, highest to lowest
                sorted_indices = list(reversed(np.argsort([box.probs[c] for box in images_boxes[image_idx]])))

                for i in range(len(sorted_indices)):
                    index_i = sorted_indices[i]

                    if images_boxes[image_idx][index_i].probs[c] != 0:
                        # for all the subsequent images_boxes[image_idx] (index j), which will have lower probability on the same class
                        for j in range(i + 1, len(sorted_indices)):
                            index_j = sorted_indices[j]

                            # We suppress only if the class is the same
                            # if the iou of a box with a lower probability (descending order) is very high, remove that box
                            if images_boxes[image_idx][index_i].class_type == images_boxes[image_idx][index_j].class_type:
                                if images_boxes[image_idx][index_i].iou(images_boxes[image_idx][index_j]) > iou_threshold:
                                    images_boxes[image_idx][index_i].probs[c] = 0

        return images_boxes

    def non_max_suppr_discard_small_ones(self, images_boxes):

        n_classes = self.parameters.n_classes
        iou_threshold = self.parameters.iou_threshold
        raise RuntimeError("No longer implemented correctly, identical to the other version")
        for image_idx in range(len(images_boxes)):

            for c in range(n_classes):
                # for each class get a list of indices that allow to access the images_boxes[image_idx] in the
                # order of probability per class, highest to lowest
                sorted_indices = list(reversed(np.argsort([box.probs[c] for box in images_boxes[image_idx]])))

                for i in range(len(sorted_indices)):
                    index_i = sorted_indices[i]

                    if images_boxes[image_idx][index_i].probs[c] != 0:
                        # for all the subsequent images_boxes[image_idx] (index j), which will have lower probability on the same class
                        for j in range(i + 1, len(sorted_indices)):
                            index_j = sorted_indices[j]

                            # if the iou of a box with a lower probability (descending order) is very high, remove that box
                            if images_boxes[image_idx][index_i].iou(images_boxes[image_idx][index_j]) > iou_threshold:
                                images_boxes[image_idx][index_j].probs[c] = 0

        return images_boxes
