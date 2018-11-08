import os
import tensorflow as tf
import numpy as np
from visualization.BoundBox import BoundBox
import cv2


class Predict(object):

    def __init__(self, parameters, preprocessor, network):
        self.parameters = parameters
        self.preprocessor = preprocessor
        self.network_loaded = False
        self.network = network

    def network_output_pipeline(self, image, pure_cv2_image, train_sess=None):
        # image_to_visualize = self.preprocessor.read_image(image_path=image_path)

        if self.parameters.training is False:
            network_output = self.run_network(image=image, pure_cv2_image=pure_cv2_image)
        else:
            network_output = self.run_net_in_training(image=image, pure_cv2_image=pure_cv2_image,
                                                      train_sess=train_sess)

        output_boxes = self.get_output_boxes(netout=network_output)
        output_boxes = self.non_max_suppression(boxes=output_boxes)
        boxes_to_print = self.get_final_boxes(image=image,
                                              boxes=output_boxes)
        return boxes_to_print

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
            self.output_node, self.image, self.train_flag =  self.network.get_network()
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

    def run_net_in_training(self, image, pure_cv2_image, train_sess):
        graph = tf.get_default_graph()
        image_ph = graph.get_tensor_by_name("image_placeholder:0")
        train_flag_ph = graph.get_tensor_by_name("flag_placeholder:0")
        output_node_ph = graph.get_tensor_by_name('network_output:0')

        image_to_network = self.preprocessor.preprocess_for_inference(image=image, pure_cv2_image=pure_cv2_image)

        network_output = train_sess.run(fetches=output_node_ph,
                                           feed_dict={image_ph: [image_to_network],
                                                      train_flag_ph: False})
        return network_output

    def sigmoid(self, x):
        x = 1. / (1. + np.exp(-x))
        return x

    def softmax(self, x):
        x = np.exp(x) / np.sum(np.exp(x), axis=0)
        return x

    def get_output_boxes(self, netout):
        output_h = self.parameters.output_h
        output_w = self.parameters.output_w
        n_anchors = self.parameters.n_anchors
        n_classes = self.parameters.n_classes
        anchors = self.parameters.anchors
        threshold = self.parameters.threshold

        boxes = []

        # interpret the output by the network
        for row in range(output_h):
            for col in range(output_w):
                for b in range(n_anchors):
                    box = BoundBox(class_num=n_classes, accuracy_mode=False)
                    # first 5 values for x, y, w, h and confidence
                    box.x, box.y, box.w, box.h, box.c = netout[0, row, col, b, :5]

                    # calculate the center and shift to dimension 1 (divide by grid)
                    box.x = (col + self.sigmoid(box.x)) / output_w
                    box.y = (row + self.sigmoid(box.y)) / output_h
                    # calculate the width and height and go to dimension 1
                    box.w = anchors[2 * b + 0] * np.exp(box.w) / output_w
                    box.h = anchors[2 * b + 1] * np.exp(box.h) / output_h
                    # calculate confidence
                    box.c = self.sigmoid(box.c)
                    # rest of values for class likelihoods
                    classes = netout[0, row, col, b, 5:]
                    # multiply probabilities per confidence
                    box.probs = self.softmax(classes) * box.c
                    box.probs *= box.probs >= threshold  # change this!

                    boxes.append(box)

        return boxes

    def non_max_suppression(self, boxes):

        n_classes = self.parameters.n_classes
        iou_threshold = self.parameters.iou_threshold

        for c in range(n_classes):
            # for each class get a list of indices that allow to access the boxes in the
            # order of probability per class, highest to lowest
            sorted_indices = list(reversed(np.argsort([box.probs[c] for box in boxes])))

            for i in range(len(sorted_indices)):
                index_i = sorted_indices[i]

                if boxes[index_i].probs[c] != 0:
                    # for all the subsequent boxes (index j), which will have lower probability on the same class
                    for j in range(i + 1, len(sorted_indices)):
                        index_j = sorted_indices[j]

                        # if the iou of a box with a lower probability (descending order) is very high, remove that box
                        if self.parameters.keep_small_ones is True:
                            if boxes[index_i].iou(boxes[index_j]) > iou_threshold and boxes[index_j].probs[c] > self.parameters.threshold and boxes[index_j].area() < boxes[index_i].area():
                                boxes[index_i].probs[c] = 0
                        else:
                            if boxes[index_i].iou(boxes[index_j]) > iou_threshold:
                                boxes[index_j].probs[c] = 0
                        # add by me: remove boxes inside the another box with small area
                        if boxes[index_i].is_matrioska(boxes[index_j]):
                            boxes[index_j].probs[c] = 0

        return boxes

    def object_area(self, obj):
        width = obj['xmax'] - obj['xmin']
        height = obj['ymax'] - obj['ymin']
        area = width * height
        return area

    def get_final_boxes(self, image, boxes):
        threshold = self.parameters.threshold
        #labels = {'0': 'yellow_cones', '1': 'blue_cones', '2': 'orange_cones'}

        output_boxes = []

        for box in boxes:
            # one one class prediction per box
            max_indx = np.argmax(box.probs)     # get class

            # one box per class
            max_prob = box.probs[max_indx]
            obj_label = self.parameters.labels_list[max_indx]    # labels[str(max_indx)]

            if max_prob >= threshold:
                # print(max_prob)
                xmin = int((box.x - box.w / 2) * image.shape[1])
                xmax = int((box.x + box.w / 2) * image.shape[1])
                ymin = int((box.y - box.h / 2) * image.shape[0])
                ymax = int((box.y + box.h / 2) * image.shape[0])
                output_box = {'xmin': xmin, 'ymin': ymin, 'xmax': xmax, 'ymax': ymax,
                              'name': obj_label, 'prob': max_prob}
                output_boxes.append(output_box)

        return output_boxes
