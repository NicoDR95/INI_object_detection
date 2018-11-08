from visualization.BoundBox import BoundBox
import numpy as np
import logging
import cv2
import tensorflow as tf
log = logging.getLogger()


class Accuracy(object):

    def __init__(self, parameters, prediction, dataset, preprocessor, visualize):
        self.parameters = parameters
        self.prediction = prediction
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.visualize = visualize

    def run_and_get_accuracy(self, train_sess=None, step=None):
        validation_images_dir = self.parameters.validation_images_dir
        validation_annotations_dir = self.parameters.validation_annotations_dir
        thr = self.parameters.threshold
        iou_accuracy_thr = self.parameters.iou_accuracy_thr
        n_classes = self.parameters.n_classes
        training = self.parameters.training
        fsg_accuracy_mode = self.parameters.fsg_accuracy_mode
        valid_set_anns = self.dataset.get_dataset_dict(validation_annotations_dir)

        all_precisions = []
        all_recalls = []
        all_F1_scores = []

        overall_true_positives = 0
        overall_pred_obj = 0
        overall_real_obj = 0


        print("Processing images...")
        for image_entry in valid_set_anns:

            image_path = validation_images_dir + image_entry['filename']
            # Ground truth is a list of dicts, each is a cone
            ground_truth = image_entry['object']

            # Returns a list of dicts, each is a cone
            image, pure_cv2_image = self.preprocessor.read_image(image_path=image_path)

            if training is False and fsg_accuracy_mode is False:
                net_output = self.prediction.network_output_pipeline(image=image, pure_cv2_image=pure_cv2_image)
            elif training is True and fsg_accuracy_mode is False:
                net_output = self.prediction.network_output_pipeline(image=image, pure_cv2_image=pure_cv2_image, train_sess=train_sess)
            else:
                net_output = self.get_fsg_output(image_name=image_entry['filename'], image=image)

            if self.parameters.visualize_accuracy_outputs is True and training is False:
                self.visualize_accuracy_output(image, net_output)

            # print(net_output)
            # print(ground_truth)

            n_obj_true = len(ground_truth)
            n_obj_pred = len(net_output)

            # print("In image {} there are {} objects".format(image_entry['filename'], n_obj_true))
            # print("The network with threshold={} and IoU_threshold={} predicted {} objects".format(thr, iou_thr, n_obj_pred))

            # print(n_obj_true)
            # print(n_obj_pred)
            # precision = TP/(TP+FP)
            # recall = TP/(TP+FN)
            # true_positives + false_positive = tot_pred_obj
            # true_positives + false_negative = tot_real_obj

            true_positives = 0
            tot_pred_obj = n_obj_pred
            tot_real_obj = n_obj_true

            for pred_obj in net_output:
                pred_box = BoundBox(n_classes, accuracy_mode=True)
                pred_box.xmin = pred_obj['xmin']
                pred_box.xmax = pred_obj['xmax']
                pred_box.ymin = pred_obj['ymin']
                pred_box.ymax = pred_obj['ymax']
                pred_box.class_type = pred_obj['name']

                for real_obj in ground_truth:
                    real_box = BoundBox(n_classes, accuracy_mode=True)
                    real_box.xmin = real_obj['xmin']
                    real_box.xmax = real_obj['xmax']
                    real_box.ymin = real_obj['ymin']
                    real_box.ymax = real_obj['ymax']
                    real_box.class_type = real_obj['name']

                    iou_pred_real = pred_box.iou(real_box, accuracy_mode=True)


                    if iou_pred_real > iou_accuracy_thr:

                        try:
                            if real_obj['matched'] is False and pred_obj['matched'] is False:
                                real_obj['matched'] = True
                                true_positives += 1
                        except KeyError:
                            real_obj['matched'] = True
                            pred_obj['matched'] = True
                            true_positives += 1

                            break

            # Calculate per image metrics
            try:
                image_precision = true_positives / tot_pred_obj
            except ZeroDivisionError:
                if tot_pred_obj == 0 and tot_real_obj == 0:
                    image_precision = 1
                else:
                    image_precision = 0
            try:
                image_recall = true_positives / tot_real_obj
            except ZeroDivisionError:
                if tot_real_obj == 0 and tot_real_obj == 0:
                    image_recall = 1
                else:
                    image_recall = 0
            try:
                image_F1_score = 2*image_precision*image_recall/(image_precision + image_recall)
            except ZeroDivisionError:
                if tot_real_obj == 0 and tot_real_obj == 0:
                    image_F1_score = 1
                else:
                    image_F1_score = 0

            # Store per image metrics
            all_precisions.append(image_precision)
            all_recalls.append(image_recall)
            all_F1_scores.append(image_F1_score)

            # Store overall values for overall metrics
            overall_true_positives += true_positives
            overall_pred_obj += tot_pred_obj
            overall_real_obj += tot_real_obj

            assert(0 <= image_precision <= 1)
            assert(0 <= image_recall <= 1)
            assert(0 <= image_F1_score <= 1)

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
            overall_F1_score = 2*overall_precision*overall_recall/(overall_precision + overall_recall)
        except ZeroDivisionError:
            overall_F1_score = 0


        log.info("There are {} images in the validation set".format(len(valid_set_anns)))
        log.info("The following values are obtained with threshold={}, IoU_threshold={} IoU_accuracy_threshold={}".format(thr, self.parameters.iou_threshold, iou_accuracy_thr))
        log.info("The mean precision is: {}".format(mean_precision))
        log.info("The mean recall is:    {}".format(mean_recall))
        log.info("The mean F1 score is:  {} ".format(mean_F1_score))

        log.info("\n \nThe following values are overall values")
        log.info("obtained calculating precision and recall on all the images at once")
        log.info("The overall precision is: {}".format(overall_precision))
        log.info("The overall recall is:    {}".format(overall_recall))
        log.info("The overall F1 score is:  {}".format(overall_F1_score))

        if training is True:
            summary_writer = tf.summary.FileWriter(self.parameters.tensorboard_dir)
            summary = tf.Summary()
            summary.value.add(tag='mean_precision', simple_value=mean_precision)
            summary.value.add(tag='mean_recall', simple_value=mean_recall)
            summary.value.add(tag='mean_F1_score', simple_value=mean_F1_score)
            summary.value.add(tag='overall_precision', simple_value=overall_precision)
            summary.value.add(tag='overall_recall', simple_value=overall_recall)
            summary.value.add(tag='overall_F1_score', simple_value=overall_F1_score)
            summary_writer.add_summary(summary, step)
            summary_writer.flush()

# TODO implement: if 2 predictions are trying to predict the same object only one should count as true positive
# todo, the others as false positive

    def get_fsg_output(self, image_name, image):
        fsg_output_dir = '/home/nico/semester_project/FSG_YOLO/pytorch-yolo2-marvis/outputs/'

        out_file = image_name[-10:-4] + '.npy'
        output_path = fsg_output_dir + out_file
        output = np.load(output_path)                    # (1, 50, 9, 13)
        # print(output)
        network_output = output.swapaxes(1, 3)           # (1, 13, 9, 50)
        network_output = network_output.swapaxes(1, 2)   # (1, 9, 13, 50)
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