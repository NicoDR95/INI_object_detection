import logging
from copy import deepcopy
from math import floor

import numpy as np
import scipy
import scipy.misc
import cv2

log = logging.getLogger()


class DataPreprocessing(object):
    def __init__(self, parameters):
        self.parameters = parameters

    attribute_error_string = \
        '''Found error in value of attribute {} in image {}. Value is {}, constraint is {}
        Object: {}
        '''

    compare_error_string = \
        '''Found error in compare of attributes {} vs {} in image {}. Values are {} vs {}
        Object: {}
        '''

    def read_image(self, image_path, image_n=None):

        # image = scipy.ndimage.imread(image_path, mode='RGB')

        # To clean the dataset use:
        # try:
        #     image = scipy.ndimage.imread(image_path, mode='RGB')
        # except FileNotFoundError:
        #     image = scipy.ndimage.imread('/home/nico/semester_project/cone_detector_data/dataset/cones_dataset2018/bad_images/' + image_n, mode='RGB') # todo, remove this
        # # image_height, image_width, image_channels = self.image.shape

        pure_cv2_image = cv2.imread(image_path)

        if pure_cv2_image is None:
            return None, None
        else:
            # if self.parameters.add_fourth_channel is True:
            #     self.add_fourth_channel(image)
            image = cv2.cvtColor(pure_cv2_image, cv2.COLOR_BGR2RGB)

            return image, pure_cv2_image


    def read_image_and_xml(self, image_annotation):
        image_annotation = image_annotation
        image_objects = image_annotation['object'][:]
        filename = image_annotation['filename']

        iteration = 0
        try:
            while True:

                image_path = self.parameters.all_images_dir[iteration] + filename
                image, pure_cv2_image = self.read_image(image_path, image_n=filename)
                iteration += 1
                if image is not None:
                    break
        except IndexError:
            log.error("Image {} not found in any of the specified images folder".format(filename))

        # In the objects we keep the filename to use as a flag to allow
        # successive object/image matching for debug purposes

        return image, pure_cv2_image, image_objects, image_annotation['filename']

    def resize_image(self, image_to_resize):

        # resized_image = scipy.misc.imresize(image_to_resize, (self.parameters.input_h, self.parameters.input_w))
        resized_image = cv2.resize(image_to_resize, (self.parameters.input_w, self.parameters.input_h))

        resized_image = np.array(resized_image, dtype=np.float32)
        return resized_image

    def resize_image_and_boxes(self, image_to_resize, image_objects, filename):
        image_height, image_width, image_channels = image_to_resize.shape

        resized_image = self.resize_image(image_to_resize=image_to_resize)

        image_height = float(image_height)
        image_width = float(image_width)
        input_h = float(self.parameters.input_h)
        input_w = float(self.parameters.input_w)
        output_w = float(self.parameters.output_w)
        output_h = float(self.parameters.output_h)

        netin = "_netin"
        oneb = "_oneb"
        xmin_netin = "xmin" + netin
        ymin_netin = 'ymin' + netin
        xmax_netin = "xmax" + netin
        ymax_netin = 'ymax' + netin

        xmin_oneb = "xmin" + oneb
        ymin_oneb = 'ymin' + oneb
        xmax_oneb = "xmax" + oneb
        ymax_oneb = 'ymax' + oneb

        # Values not rounded to INT to allow training on floating point values
        #  and eventually original value reconstruction
        for obj in image_objects:

            # center in original space
            obj["xcenter_orig"] = 0.5 * (obj['xmin'] + obj['xmax'])
            obj["ycenter_orig"] = 0.5 * (obj['ymin'] + obj['ymax'])

            # center in grid space
            # Moved before x y rescaling to minimize numerical error
            obj["x_grid"] = obj["xcenter_orig"] * output_w / image_width
            obj["y_grid"] = obj["ycenter_orig"] * output_h / image_height

            # todo, remove the comment
            assert (0 <= obj["x_grid"] < output_w), self.attribute_error_string.format("x_grid", filename, obj["x_grid"], output_w, obj)
            assert (0 <= obj["y_grid"] < output_h), self.attribute_error_string.format("y_grid", filename, obj["y_grid"], output_w, obj)

            obj["x_grid_rel"] = obj["x_grid"] - floor(obj["x_grid"])
            obj["y_grid_rel"] = obj["y_grid"] - floor(obj["y_grid"])

            assert (0 <= obj["x_grid_rel"] <= 1)
            assert (0 <= obj["y_grid_rel"] <= 1)

            if output_h < input_h:
                assert (obj["x_grid"] <= obj["xcenter_orig"]), self.compare_error_string.format("x_grid", 'xcenter_orig', obj['x_grid'],
                                                                                                obj['xcenter_orig'], obj)

            if output_w < output_h:
                assert (obj["y_grid"] <= obj["ycenter_orig"]), self.compare_error_string.format("y_grid", 'ycenter_orig', obj['y_grid'],
                                                                                                obj['ycenter_orig'], obj)

            for attr in ['xmin', 'xmax']:
                obj[attr + oneb] = obj[attr] / image_width
                obj[attr + netin] = obj[attr + oneb] * input_w
                try:
                    assert (0 <= obj[attr + netin] <= input_w)
                except AssertionError:
                    log.error(self.attribute_error_string.format(attr, filename, obj[attr + netin], input_w, obj))

            for attr in ['ymin', 'ymax']:
                obj[attr + oneb] = obj[attr] / image_height
                obj[attr + netin] = obj[attr + oneb] * input_h

                try:
                    assert (0 <= obj[attr + netin] <= input_h)
                except AssertionError:
                    log.error(self.attribute_error_string.format(attr, filename, obj[attr + netin], input_h, obj))

            obj["box"] = [obj['xmin_oneb'], obj['ymin_oneb'], obj['xmax_oneb'], obj['ymax_oneb']]

            assert (obj[xmin_netin] <= obj[xmax_netin]), self.compare_error_string.format(xmin_netin, xmax_netin, obj[xmin_netin],
                                                                                         obj[xmax_netin], obj)
            assert (obj[ymin_netin] <= obj[ymax_netin]), self.compare_error_string.format(ymin_netin, ymax_netin, obj[ymin_netin],
                                                                                         obj[ymax_netin], obj)
            assert (obj[xmin_oneb] <= obj[xmax_oneb]), self.compare_error_string.format(xmin_oneb, xmax_oneb,
                                                                                        obj[xmin_oneb],
                                                                                        obj[xmax_oneb], obj)
            assert (obj[ymin_oneb] <= obj[ymax_oneb]), self.compare_error_string.format(ymin_oneb, ymax_oneb,
                                                                                        obj[ymin_oneb],
                                                                                        obj[ymax_oneb], obj)

        return resized_image, image_objects

    def normalize(self, image_to_normalize):

        normalized_image = image_to_normalize / self.parameters.data_preprocessing_normalize

        return normalized_image

    def preprocess_for_training(self, image):
        image, pure_cv2_image, objects, filename = self.read_image_and_xml(image)
        image, objects = self.resize_image_and_boxes(image, objects, filename)
        image = self.normalize(image)
        if self.parameters.add_fourth_channel is True:
            image = self.add_fourth_channel(pure_image=pure_cv2_image, preprocessed_image=image)

        return image, objects, filename

    def preprocess_for_inference(self, image, pure_cv2_image):
        # image = self.read_image(image_path=image_path)
        image = self.resize_image(image_to_resize=image)

        image = self.normalize(image_to_normalize=image)
        if self.parameters.add_fourth_channel is True:
            image = self.add_fourth_channel(pure_image=pure_cv2_image, preprocessed_image=image)
        return image

    def add_fourth_channel(self, pure_image, preprocessed_image):

        # Remember cv2 work in BGR
        # We'll work in HSV space
        # denormilized_image = image * self.parameters.data_preprocessing_normalize

        # bgr_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        image_hsv = cv2.cvtColor(pure_image, cv2.COLOR_BGR2HSV)

        # blue range values
        # blue HSV: [120 255 255]
        lower_blue = np.array([90, 20, 20])
        upper_blue = np.array([140, 255, 255])
        fourth_chan_blue = cv2.inRange(image_hsv, lower_blue, upper_blue)

        # Yellow range values
        # Yellow HSV: [30 255 255]
        lower_yellow = np.array([15, 55, 80])
        upper_yellow = np.array([35, 255, 255])
        fourth_chan_yellow = cv2.inRange(image_hsv, lower_yellow, upper_yellow)

        # orange range values
        # orange HSV: [[[8 255 255]]]
        lower_orange = np.array([2, 55, 60])
        upper_orange = np.array([15, 255, 255])
        fourth_chan_orange = cv2.inRange(image_hsv, lower_orange, upper_orange)

        # bitwise or to gain the full result
        full_fourth_channel = cv2.bitwise_or(fourth_chan_blue, fourth_chan_yellow)
        full_fourth_channel = cv2.bitwise_or(full_fourth_channel, fourth_chan_orange)
        filtered_fourth_channel = cv2.medianBlur(full_fourth_channel, 5)
        fourth_channel = filtered_fourth_channel
        if self.parameters.use_grayscale_mask is True:
            grayscale_pure_image = cv2.cvtColor(pure_image, cv2.COLOR_BGR2GRAY)
            gray_scale_fourth_channel = cv2.bitwise_and(grayscale_pure_image, grayscale_pure_image, mask=filtered_fourth_channel)
            fourth_channel = gray_scale_fourth_channel/self.parameters.data_preprocessing_normalize
        if self.parameters.use_hue_mask is True:
            hsv_fourth_channel = cv2.bitwise_and(image_hsv, image_hsv, mask=filtered_fourth_channel)
            hue_fourth_channel, _, _ = cv2.split(hsv_fourth_channel)
            fourth_channel = hue_fourth_channel/self.parameters.data_preprocessing_normalize

        resized_fourth_channel = self.resize_image(fourth_channel)

        if self.parameters.visualize_fourth_channel is True:
            cv2.imshow('image', pure_image)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow('fourth_channel', full_fourth_channel)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow('median', filtered_fourth_channel)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow('resized', fourth_channel)
            cv2.waitKey()
            cv2.destroyAllWindows()
            cv2.imshow('resized', resized_fourth_channel)
            cv2.waitKey()
            cv2.destroyAllWindows()

        r_channel, g_channel, b_channel = cv2.split(preprocessed_image)

        four_chan_image = cv2.merge((r_channel, g_channel, b_channel, resized_fourth_channel))

        return four_chan_image

