import logging
import math

import cv2
import numpy as np
import tensorflow as tf

from visualization.BoundBox import BoundBox

log = logging.getLogger()
import time


class Accuracy(object):

    def __init__(self, parameters, prediction, datasets, preprocessor, visualize):
        self.parameters = parameters
        self.prediction = prediction
        self.datasets = datasets
        self.preprocessor = preprocessor
        self.visualize = visualize
        self.max_mean_f1_score = dict()
        self.max_overall_F1_score = dict()
        self.summary_writer = tf.summary.FileWriter(self.parameters.tensorboard_dir)
        self.batch_time = 0

    # @measure_time
    def read_images_batch(self, images, path):
        read_images = list()
        read_pure_cv2_images = list()
        read_ground_truths = list()

        for image_entry in images:
            image_path = path + image_entry['filename']
            # Ground truth is a list of boxes
            image_objects = image_entry['object']
            image_boxes = list()

            for single_object in image_objects:
                class_index = self.parameters.labels_list.index(single_object["name"])

                box = BoundBox(xmin=single_object["xmin"],
                               xmax=single_object["xmax"],
                               ymin=single_object["ymin"],
                               ymax=single_object["ymax"],
                               class_type=class_index,
                               conf=None,
                               probs=None
                               )

                image_boxes.append(box)

            read_ground_truths.append(image_boxes)
            # Returns a list of dicts, each is a cone
            image, pure_cv2_image = self.preprocessor.read_image(image_path=image_path)
            # image = self.preprocessor.normalize(image)
            read_images.append(image)
            read_pure_cv2_images.append(pure_cv2_image)

        return read_images, read_pure_cv2_images, read_ground_truths

    # @measure_time
    def run_and_get_accuracy(self, train_sess=None, step=None, epoch_finished=True):

        for dataset_dict in self.datasets:
            dataset_name = dataset_dict["dataset_name"]
            dataset  = dataset_dict["dataset"]
            log.info("\n \n \n Running accuracy on dataset {} ".format(dataset_name))
            self.run_accuracy_on_dataset(dataset_name, dataset, train_sess, step, epoch_finished)

        if self.parameters.training is True and epoch_finished is True:
            self.send_to_tf_summary_global(step)

    def run_accuracy_on_dataset(self, dataset_name, dataset, train_sess=None, step=None, epoch_finished=True):
        # images_dir = dataset.images_dir

        conf_thr = self.parameters.conf_threshold
        iou_accuracy_thr = self.parameters.iou_accuracy_thr

        training = self.parameters.training
        fsg_accuracy_mode = self.parameters.fsg_accuracy_mode
        valid_set_anns = dataset.get_dataset_dict()

        all_precisions = []
        all_recalls = []
        all_F1_scores = []

        overall_true_positives = 0
        overall_pred_obj = 0
        overall_real_obj = 0
        num_images = len(valid_set_anns)

        num_batches = math.ceil(num_images / self.parameters.batch_size)

        # print("valid_set_anns",valid_set_anns)
        # print("num_batches", num_batches)
        batch_ranges = np.linspace(0, num_images, num_batches, dtype=np.int, endpoint=False).tolist()
        batch_ranges.append(num_images)  # to ensure there is no error due to linspace roundup
        resulting_num_batches = len(batch_ranges) - 1

        log.info("Dataset {} -  Computing accuracy with {} batches (batch size = {})".format(dataset_name,
                                                                                             resulting_num_batches,
                                                                                             self.parameters.batch_size))
        self.batch_time = 0
        for batch_range_idx in range(resulting_num_batches):
            images = valid_set_anns[batch_ranges[batch_range_idx]:batch_ranges[batch_range_idx + 1]]

            batch_precisions, batch_recalls, batch_F1_scores, batch_true_positives, batch_tot_pred_obj, batch_tot_real_obj = self.process_batch(
                images,
                training,
                fsg_accuracy_mode,
                train_sess,
                dataset.images_path,
                iou_accuracy_thr)

            all_precisions.extend(batch_precisions)
            all_recalls.extend(batch_recalls)
            all_F1_scores.extend(batch_F1_scores)

            overall_true_positives = overall_true_positives + batch_true_positives
            overall_pred_obj = overall_pred_obj + batch_tot_pred_obj
            overall_real_obj = overall_real_obj + batch_tot_real_obj

        # calculate mean of results on all images
        mean_precision = np.mean(all_precisions)
        mean_recall = np.mean(all_recalls)
        mean_F1_score = np.mean(all_F1_scores)

        # pytorch alike way
        try:
            overall_precision = overall_true_positives / overall_pred_obj
        except ZeroDivisionError:
            if overall_pred_obj == 0 and overall_real_obj == 0:
                overall_precision = 1
            else:
                overall_precision = 0
        try:
            overall_recall = overall_true_positives / overall_real_obj
        except ZeroDivisionError:
            if overall_pred_obj == 0 and overall_real_obj == 0:
                overall_recall = 1
            else:
                overall_recall = 0
        try:
            overall_F1_score = 2 * overall_precision * overall_recall / (overall_precision + overall_recall)
        except ZeroDivisionError:
            overall_F1_score = 0

        frame_time = self.batch_time / num_images
        fps = 1 / frame_time

        log.info("There are {} images in the validation set".format(len(valid_set_anns)))
        log.info("The following values are obtained with conf_threshold={}, IoU_threshold={} IoU_accuracy_threshold={}".format(
            conf_thr,
            self.parameters.iou_threshold,
            iou_accuracy_thr))

        log.info("There were {} objects in dataset {}, total predictions are {} and {} are correct".format(overall_real_obj,
                                                                                                           dataset_name,
                                                                                                           overall_pred_obj,
                                                                                                           overall_true_positives))
        log.info("The mean precision is: {}".format(mean_precision))
        log.info("The mean recall is:    {}".format(mean_recall))
        log.info("The mean F1 score is:  {} ".format(mean_F1_score))

        log.info("\n \nThe following values are overall values")
        log.info("obtained calculating precision and recall on all the images at once")
        log.info("The overall precision is: {}".format(overall_precision))
        log.info("The overall recall is:    {}".format(overall_recall))
        log.info("The overall F1 score is:  {}".format(overall_F1_score))
        log.info("Validation required {} overall, {:.2f} per frame, {:.2f} FPS".format(self.batch_time, frame_time, fps))

        if training is True and epoch_finished is True:
            self.send_to_tf_summary_dataset(step, mean_precision, mean_recall, mean_F1_score, overall_precision, overall_recall, overall_F1_score, dataset_name)

    # @measure_time
    def send_to_tf_summary_dataset(self, step, mean_precision, mean_recall, mean_F1_score, overall_precision, overall_recall, overall_F1_score, dataset_name):
        try:
            if self.max_mean_f1_score["dataset_name"] < mean_F1_score:
                self.max_mean_f1_scoree["dataset_name"] = mean_F1_score
        except KeyError:
            self.max_mean_f1_scoree["dataset_name"] = mean_F1_score

        try:
            if self.max_overall_F1_scoree["dataset_name"] < overall_F1_score:
                self.max_overall_F1_scoree["dataset_name"] = overall_F1_score
        except KeyError:
            self.max_overall_F1_scoree["dataset_name"] = overall_F1_score

        summary = tf.Summary()

        summary.value.add(tag=dataset_name + " " + 'mean_precision', simple_value=mean_precision)
        summary.value.add(tag=dataset_name + " " + 'mean_recall', simple_value=mean_recall)
        summary.value.add(tag=dataset_name + " " + 'mean_F1_score', simple_value=mean_F1_score)
        summary.value.add(tag=dataset_name + " " + 'max_mean_f1_score', simple_value=self.max_mean_f1_score["dataset_name"])

        summary.value.add(tag=dataset_name + " " + 'overall_precision', simple_value=overall_precision)
        summary.value.add(tag=dataset_name + " " + 'overall_recall', simple_value=overall_recall)
        summary.value.add(tag=dataset_name + " " + 'overall_F1_score', simple_value=overall_F1_score)
        summary.value.add(tag=dataset_name + " " + 'max_overall_F1_score', simple_value=self.max_overall_F1_score["dataset_name"])

        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()

    def send_to_tf_summary_global(self, step):

        summary = tf.Summary()

        summary.value.add(tag='Batch_size vs epochs', simple_value=self.parameters.batch_size)
        summary.value.add(tag='Learning rate vs epochs', simple_value=self.parameters.learning_rate)

        self.summary_writer.add_summary(summary, step)
        self.summary_writer.flush()

    # @measure_time
    def process_batch(self, images, training, fsg_accuracy_mode, train_sess, path, iou_accuracy_thr):

        read_images, read_pure_cv2_images, read_ground_truths = self.read_images_batch(images=images, path=path)

        if training is True:
            batch_time_start = time.time()
            net_output_batch = self.prediction.network_output_pipeline(images=read_images,
                                                                       pure_cv2_images=read_pure_cv2_images,
                                                                       train_sess=train_sess)
            self.batch_time = time.time() + self.batch_time - batch_time_start
        elif fsg_accuracy_mode is False:
            batch_time_start = time.time()
            net_output_batch = self.prediction.network_output_pipeline(images=read_images,
                                                                       pure_cv2_images=read_pure_cv2_images)
            self.batch_time = time.time() + self.batch_time - batch_time_start
        else:
            if self.parameters.batch_size != 1:
                raise AttributeError("FSG mode supports only batch size of 1")
            image = read_images[0]
            image_entry = images[0]
            net_output_batch = self.get_fsg_output(image_name=image_entry['filename'], image=image)

        if self.parameters.visualize_accuracy_outputs is True and training is False:
            if self.parameters.batch_size != 1:
                log.warn("Batch size larger than 1 in visualization mode, only first batch image visualized")
            image = read_images[0]
            net_output = net_output_batch[0]
            batch_time_start = time.time()
            self.visualize_accuracy_output(image, net_output)
            self.batch_time = time.time() + self.batch_time - batch_time_start

        batch_precisions = list()
        batch_recalls = list()
        batch_F1_scores = list()
        true_positives_batch = 0
        tot_pred_obj_batch = 0
        tot_real_obj_batch = 0

        for image_idx, net_output in enumerate(net_output_batch):

            image_ground_truth = read_ground_truths[image_idx]

            true_positives_image = 0
            img_pred_obj = len(net_output)
            img_real_obj = len(image_ground_truth)

            tot_pred_obj_batch = tot_pred_obj_batch + img_pred_obj
            tot_real_obj_batch = tot_real_obj_batch + img_real_obj

            unmatched_pred_indexes = list(range(len(net_output)))

            # for pp in net_output:
            #    print(pp.conf)

            for real_box in image_ground_truth:

                for list_idx, pred_box_idx in enumerate(unmatched_pred_indexes):

                    pred_box = net_output[pred_box_idx]

                    if pred_box.class_type == real_box.class_type:
                        iou_pred_real = pred_box.iou(real_box)

                        if iou_pred_real >= iou_accuracy_thr:
                            true_positives_image = true_positives_image + 1
                            # we break so it is fine to pop directly on the list
                            unmatched_pred_indexes.pop(list_idx)
                            break

            assert (true_positives_image <= len(image_ground_truth))
            assert (true_positives_image <= len(net_output))

            true_positives_batch = true_positives_batch + true_positives_image
            # Calculate per image metrics
            try:
                image_precision = true_positives_image / img_pred_obj
            except ZeroDivisionError:
                if img_pred_obj == 0 and img_real_obj == 0:
                    image_precision = 1
                else:
                    image_precision = 0
            try:
                image_recall = true_positives_image / img_real_obj
            except ZeroDivisionError:
                if img_pred_obj == 0 and img_real_obj == 0:
                    image_recall = 1
                else:
                    image_recall = 0
            try:
                image_F1_score = 2 * image_precision * image_recall / (image_precision + image_recall)
            except ZeroDivisionError:
                if img_pred_obj == 0 and img_real_obj == 0:
                    image_F1_score = 1
                else:
                    image_F1_score = 0

            # Store per image metrics
            batch_precisions.append(image_precision)
            batch_recalls.append(image_recall)
            batch_F1_scores.append(image_F1_score)

            assert (0 <= image_precision <= 1)
            assert (0 <= image_recall <= 1)
            assert (0 <= image_F1_score <= 1)
        return batch_precisions, batch_recalls, batch_F1_scores, true_positives_batch, tot_pred_obj_batch, tot_real_obj_batch

    # TODO implement: if 2 predictions are trying to predict the same object only one should count as true positive
    # todo, the others as false positive

    def get_fsg_output(self, image_name, image):
        fsg_output_dir = '/home/nico/semester_project/FSG_YOLO/pytorch-yolo2-marvis/outputs/'

        out_file = image_name[-10:-4] + '.npy'
        output_path = fsg_output_dir + out_file
        output = np.load(output_path)  # (1, 50, 9, 13)
        # print(output)
        network_output = output.swapaxes(1, 3)  # (1, 13, 9, 50)
        network_output = network_output.swapaxes(1, 2)  # (1, 9, 13, 50)
        network_output = np.reshape(network_output, (1, 9, 13, 5, 10))  # (1, 9, 13, 5, 10)
        # print(network_output)

        output_boxes = self.prediction.get_output_boxes(netout=network_output)
        output_boxes = self.prediction.non_max_suppression(boxes=output_boxes)
        boxes_to_print = self.prediction.get_final_boxes(image=image, boxes=output_boxes)

        if self.parameters.visualize_accuracy_outputs is True:
            self.visualize_accuracy_output(image, boxes_to_print)

        return boxes_to_print

    def visualize_accuracy_output(self, image, boxes_to_print):
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        image = image / self.parameters.data_preprocessing_normalize

        output_image = self.visualize.draw_boxes_on_image(image=image,
                                                          boxes_to_print=boxes_to_print)
        cv2.imshow('img', output_image)
        cv2.waitKey()
