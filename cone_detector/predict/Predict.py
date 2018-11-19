import numpy as np
import tensorflow as tf

from utility.utility_library import measure_time
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

        self.anchors_h = np.array(anchors_h)
        self.anchors_w = np.array(anchors_w)

    #@measure_time
    def network_output_pipeline(self, images, pure_cv2_images, train_sess=None):
        # image_to_visualize = self.preprocessor.read_image(image_path=image_path)

        if self.parameters.training is False:
            network_output = self.run_network(images=images, pure_cv2_images=pure_cv2_images)
        else:
            network_output = self.run_net_in_training(images=images, pure_cv2_images=pure_cv2_images,
                                                      train_sess=train_sess)

        output_boxes = self.get_output_boxes(net_output=network_output, image_shape_x=images[0].shape[1], image_shape_y=images[0].shape[0], )
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
    def run_network(self, image, pure_cv2_image):

        if self.network_loaded is False:
            self.load_network()

        image_to_network = self.preprocessor.preprocess_for_inference(image=image, pure_cv2_image=pure_cv2_image)

        # image_to_see = cv2.cvtColor(image_to_network, cv2.COLOR_BGR2RGB)
        # cv2.imshow('image',image_to_see)
        # cv2.waitKey()

        network_output = self.sess.run(fetches=self.output_node,
                                       feed_dict={self.image: [image_to_network],
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
        x = exp_x / np.sum(exp_x, axis=0)
        return x

    #@measure_time
    def get_output_boxes(self, net_output, image_shape_x, image_shape_y, ):

        output_h = self.parameters.output_h
        output_w = self.parameters.output_w
        n_anchors = self.parameters.n_anchors
        threshold = self.parameters.threshold
        boxes_for_all_images = list()

        boxes_x_y_sig = self.sigmoid(net_output[:, :, :, :, :2])
        boxes_w_h_exp = np.exp(net_output[:, :, :, :, 2:4])
        boxes_w = boxes_w_h_exp[:, :, :, :, 0] / output_w
        boxes_h = boxes_w_h_exp[:, :, :, :, 1] / output_h
        boxes_c_sig = self.sigmoid(net_output[:, :, :, :, 4])
        boxes_classes_soft = self.softmax(net_output[:, :, :, :, 5:])

        for col in range(output_w):
            boxes_x_y_sig[:, :, col, :, 0] = (boxes_x_y_sig[:, :, col, :, 0] + col) / output_w

        for row in range(output_h):
            boxes_x_y_sig[:, row, :, :, 1] = (boxes_x_y_sig[:, row, :, :, 1] + row) / output_h

        for image_idx in range(len(net_output)):
            boxes = []

            # interpret the output by the network
            for row in range(output_h):
                for col in range(output_w):
                    for b in range(n_anchors):
                        # first 5 values for x, y, w, h and confidence
                        conf = boxes_c_sig[image_idx, row, col, b]
                        probs = boxes_classes_soft[image_idx, row, col, b] * conf

                        max_prob = np.amax(probs)
                        # filter them out if we are not going to use them anyway
                        if max_prob >= threshold:
                            probs[probs < threshold] = 0
                            # calculate the center and shift to dimension 1 (divide by grid)
                            x = boxes_x_y_sig[image_idx, row, col, b, 0]
                            y = boxes_x_y_sig[image_idx, row, col, b, 1]
                            # calculate the width and height and go to dimension 1
                            w = self.anchors_w[b] * boxes_w[image_idx, row, col, b]
                            h = self.anchors_h[b] * boxes_h[image_idx, row, col, b]
                            # calculate confidence

                            # multiply probabilities per confidence

                            # one one class prediction per box
                            max_indx = np.argmax(probs)  # get class
                            obj_label = self.parameters.labels_list[max_indx]  # labels[str(max_indx)]

                            box = BoundBox(x=x, y=y, w=w, h=h, probs=probs, conf=conf, maxmin_x_rescale=image_shape_x, maxmin_y_rescale=image_shape_y,
                                           class_type=obj_label, groundtruth=False, xmin=None, xmax=None, ymin=None, ymax=None)

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

                            # if the iou of a box with a lower probability (descending order) is very high, remove that box

                            if images_boxes[image_idx][index_i].iou(images_boxes[image_idx][index_j]) > iou_threshold and \
                                    images_boxes[image_idx][index_j].probs[c] > self.parameters.threshold and \
                                    images_boxes[image_idx][index_j].area() < images_boxes[image_idx][index_i].area():
                                images_boxes[image_idx][index_i].probs[c] = 0

                            elif images_boxes[image_idx][index_i].is_matrioska(images_boxes[image_idx][index_j]):
                                # remove images_boxes[image_idx] inside the another box with small area
                                images_boxes[image_idx][index_j].probs[c] = 0

        return images_boxes

    def non_max_suppr_discard_small_ones(self, images_boxes):

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

                            # if the iou of a box with a lower probability (descending order) is very high, remove that box
                            if images_boxes[image_idx][index_i].iou(images_boxes[image_idx][index_j]) > iou_threshold:
                                images_boxes[image_idx][index_j].probs[c] = 0
                            elif images_boxes[image_idx][index_i].is_matrioska(images_boxes[image_idx][index_j]):
                                # remove images_boxes[image_idx] inside the another box with small area
                                images_boxes[image_idx][index_j].probs[c] = 0

        return images_boxes
