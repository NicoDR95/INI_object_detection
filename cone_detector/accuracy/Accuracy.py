import logging

import cv2
import numpy as np
import tensorflow as tf

from visualization.BoundBox import BoundBox

log = logging.getLogger()


class Accuracy(object):

    def __init__(self, parameters, prediction, dataset, preprocessor, visualize):
        self.parameters = parameters
        self.prediction = prediction
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.visualize = visualize

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
                box = BoundBox(x=None, y=None, w=None, h=None, probs=None, conf=None, maxmin_x_rescale=None, maxmin_y_rescale=None,
                               class_type=single_object["name"], groundtruth=True,
                               xmin=single_object["xmin"],
                               xmax=single_object["xmax"],
                               ymin=single_object["ymin"],
                               ymax=single_object["ymax"])
                image_boxes.append(box)

            read_ground_truths.append(image_boxes)
            # Returns a list of dicts, each is a cone
            image, pure_cv2_image = self.preprocessor.read_image(image_path=image_path)
            read_images.append(image)
            read_pure_cv2_images.append(pure_cv2_image)

        return read_images, read_pure_cv2_images, read_ground_truths

    # @measure_time
    def run_and_get_accuracy(self, train_sess=None, step=None, epoch_finished=True):
        validation_images_dir = self.parameters.validation_images_dir

        thr = self.parameters.threshold
        iou_accuracy_thr = self.parameters.iou_accuracy_thr
        n_classes = self.parameters.n_classes
        training = self.parameters.training
        fsg_accuracy_mode = self.parameters.fsg_accuracy_mode
        valid_set_anns = self.dataset.get_dataset_dict()

        all_precisions = []
        all_recalls = []
        all_F1_scores = []

        overall_true_positives = 0
        overall_pred_obj = 0
        overall_real_obj = 0
        num_images = len(valid_set_anns)
        num_batches = num_images / self.parameters.batch_size

        batch_ranges = np.linspace(0, num_images, num_batches, dtype=np.int)
        batch_ranges[-1] = num_images  # to ensure there is no error due to linspace roundup
        resulting_num_batches = len(batch_ranges) - 1

        log.info("Computing accuracy...")
        for batch_range_idx in range(resulting_num_batches):
            images = valid_set_anns[batch_ranges[batch_range_idx]:batch_ranges[batch_range_idx + 1]]

            batch_precisions, batch_recalls, batch_F1_scores, batch_true_positives, batch_tot_pred_obj, batch_tot_real_obj = self.process_batch(
                images,
                training,
                fsg_accuracy_mode,
                train_sess,
                validation_images_dir,
                n_classes,
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

        log.info("There are {} images in the validation set".format(len(valid_set_anns)))
        log.info("The following values are obtained with threshold={}, IoU_threshold={} IoU_accuracy_threshold={}".format(thr,
                                                                                                                          self.parameters.iou_threshold,
                                                                                                                          iou_accuracy_thr))
        log.info("The mean precision is: {}".format(mean_precision))
        log.info("The mean recall is:    {}".format(mean_recall))
        log.info("The mean F1 score is:  {} ".format(mean_F1_score))

        log.info("\n \nThe following values are overall values")
        log.info("obtained calculating precision and recall on all the images at once")
        log.info("The overall precision is: {}".format(overall_precision))
        log.info("The overall recall is:    {}".format(overall_recall))
        log.info("The overall F1 score is:  {}".format(overall_F1_score))

        if training is True and epoch_finished is True:
            self.send_to_tf_summary(step, mean_precision, mean_recall, mean_F1_score, overall_precision, overall_recall, overall_F1_score)

    # @measure_time
    def send_to_tf_summary(self, step, mean_precision, mean_recall, mean_F1_score, overall_precision, overall_recall, overall_F1_score):
        summary_writer = tf.summary.FileWriter(self.parameters.tensorboard_dir)
        summary = tf.Summary()
        summary.value.add(tag='mean_precision', simple_value=mean_precision)
        summary.value.add(tag='mean_recall', simple_value=mean_recall)
        summary.value.add(tag='mean_F1_score', simple_value=mean_F1_score)
        summary.value.add(tag='overall_precision', simple_value=overall_precision)
        summary.value.add(tag='overall_recall', simple_value=overall_recall)
        summary.value.add(tag='overall_F1_score', simple_value=overall_F1_score)
        summary.value.add(tag='Batch_size_VS_epochs', simple_value=self.parameters.batch_size)

        summary_writer.add_summary(summary, step)
        summary_writer.flush()

    # @measure_time
    def process_batch(self, images, training, fsg_accuracy_mode, train_sess, path, n_classes, iou_accuracy_thr):

        read_images, read_pure_cv2_images, read_ground_truths = self.read_images_batch(images=images, path=path)

        if training is True:
            net_output_batch = self.prediction.network_output_pipeline(images=read_images, pure_cv2_images=read_pure_cv2_images,
                                                                       train_sess=train_sess)
        elif fsg_accuracy_mode is False:
            net_output_batch = self.prediction.network_output_pipeline(images=read_images, pure_cv2_images=read_pure_cv2_images)

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
            self.visualize_accuracy_output(image, net_output)

        batch_precisions = list()
        batch_recalls = list()
        batch_F1_scores = list()
        true_positives_batch = 0
        tot_pred_obj_batch = 0
        tot_real_obj_batch = 0

        for image_idx, net_output in enumerate(net_output_batch):

            image_ground_truth = read_ground_truths[image_idx]

            true_positives_image = 0
            tot_pred_obj = len(net_output)
            tot_real_obj = len(image_ground_truth)

            tot_pred_obj_batch = tot_pred_obj_batch + tot_pred_obj
            tot_real_obj_batch = tot_real_obj_batch + tot_real_obj

            unmatched_pred_indexes = list(range(len(net_output)))

            for real_box in image_ground_truth:

                for list_idx, pred_box_idx in enumerate(unmatched_pred_indexes):
                    pred_box = net_output[pred_box_idx]

                    iou_pred_real = pred_box.iou(real_box, accuracy_mode=True)

                    if iou_pred_real > iou_accuracy_thr:
                        true_positives_image += 1
                        unmatched_pred_indexes.pop(list_idx)
                        break


            true_positives_batch = true_positives_batch + true_positives_image
            # Calculate per image metrics
            try:
                image_precision = true_positives_image / tot_pred_obj
            except ZeroDivisionError:
                if tot_pred_obj == 0 and tot_real_obj == 0:
                    image_precision = 1
                else:
                    image_precision = 0
            try:
                image_recall = true_positives_image / tot_real_obj
            except ZeroDivisionError:
                if tot_real_obj == 0 and tot_real_obj == 0:
                    image_recall = 1
                else:
                    image_recall = 0
            try:
                image_F1_score = 2 * image_precision * image_recall / (image_precision + image_recall)
            except ZeroDivisionError:
                if tot_real_obj == 0 and tot_real_obj == 0:
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
