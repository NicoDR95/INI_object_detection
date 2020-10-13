# comment before copy file on server:
import logging
import time

import colour
import cv2
import numpy as np
import os
import scipy.misc
from colour import Color

log = logging.getLogger()


class Visualization(object):

    def __init__(self, parameters, preprocessor, prediction):
        self.parameters = parameters
        self.preprocessor = preprocessor
        self.prediction = prediction
        self.labels_colors = self.get_color_to_labels(self.parameters.labels_list)

    def get_color_to_labels(self, labels_list, filter_bad_colours=True):
        # To avoid unreadable colors
        if filter_bad_colours is True:
            color_filters = ["white", "black", "salmon"]

            color_filters = [Color(color_filter) for color_filter in color_filters]

        elif filter_bad_colours is not False:
            color_filters = filter_bad_colours
        else:
            color_filters = []

        found_colors = list()
        labels_colors = list()
        for label in labels_list:
            label_components = label.split("_")
            colored_component = None

            for component in label_components:
                try:
                    colored_component = Color(component)
                    break
                except ValueError:
                    pass

            found_colors.append(colored_component)

        color_search_idx = 0
        color_search_values_unfiltered = [color_name[0] for color_name in list(colour.RGB_TO_COLOR_NAMES.values())]
        color_search_values = list()

        for search_color in color_search_values_unfiltered:
            to_filter = False

            for filter in color_filters:
                if Color(search_color) == filter:
                    to_filter = True
                    break

            if to_filter is False:
                color_search_values.append(search_color)

        for found in found_colors:
            if found is None:

                free_color_found = False
                while free_color_found is False:
                    color_candidate_key = color_search_values[color_search_idx]
                    color_candidate = Color(color_candidate_key)

                    free_color_found = True

                    for other_color in found_colors:
                        if color_candidate == other_color:
                            color_search_idx = color_search_idx + 1
                            free_color_found = False
                            break

                labels_colors.append(color_candidate)
                color_search_idx = color_search_idx + 1
            else:
                labels_colors.append(found)

            class_idx = len(labels_colors) - 1
            class_label = labels_list[class_idx]
            class_color = labels_colors[-1]
            log.info("Class {} got assigned box color {}".format(class_label, class_color))

        return labels_colors

    def visualize_img_before_preprocessing(self, image_annotation):
        training = self.parameters.training
        augmentation_mode = self.parameters.augmentation_mode
        augmented_image_dir = self.parameters.augmented_image_dir

        if training is False and augmentation_mode is True:
            image_path = augmented_image_dir + image_annotation['filename']
        else:
            image_path = self.parameters.images_dir + image_annotation['filename']

        image, _ = self.preprocessor.read_image(image_path=image_path)
        image_objects = image_annotation['object']
        print("visualizing the image {}".format(image_annotation['filename']))

        # Normalize the img before visualizing,
        # Hacky thing: cv2 will recognize the image as colored and print colored boxes
        # instead of grey scaled ones... cv2 is retarded as fuck
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image / self.parameters.data_preprocessing_normalize
        image = self.visualize_img_boxes(image, image_objects)

        # scipy.misc.imshow(image)
        cv2.namedWindow('image', cv2.WINDOW_NORMAL)
        cv2.imshow('image', image)
        cv2.waitKey()

        # To clean the datset use:
        # while True:
        #     cv2.imshow('image', image)
        #     k = cv2.waitKey(33)
        #     if k == 121:
        #         return 0
        #     if k == 110:
        #         shutil.move(self.parameters.annotations_dir + image_annotation['filename'][:-3] + 'xml',
        #                     "/home/nico/semester_project/cone_detector_data/dataset/cones_dataset2018/bad_annotations/" + image_annotation['filename'][:-3] + 'xml')
        #         shutil.move(self.parameters.images_dir + image_annotation['filename'],
        #                     "/home/nico/semester_project/cone_detector_data/dataset/cones_dataset2018/bad_images/" +
        #                     image_annotation['filename'])
        #
        #         print("image moved away")
        #
        #         return 0

    def object_area(self, box):
        width = box.xmax - box.xmin
        height = box.ymax - box.ymin
        area = width * height
        return area

    def object_distance_from_top(self, box):
        centery = box.ymax - (box.ymax - box.ymin) / 2
        return centery

    def colored_rectangle_writer(self, image, box):
        # color are selceted in BGR way (instead of RGB), because cv2 is retarded and works with these scheme

        box_color = self.labels_colors[box.class_type]
        box_color_rgb = box_color.rgb
        box_color_bgr = list(reversed(box_color_rgb))

        cv2.rectangle(image, (box.xmin, box.ymin), (box.xmax, box.ymax), box_color_bgr, 2)

        box_label = self.parameters.labels_list[box.class_type]
        box_text = "{} ({})".format(box_label, str(round(box.max_prob, 3)))

        cv2.putText(image, box_text, (box.xmin, box.ymin - 12), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.001 * image.shape[0],
                    color=box_color_bgr, thickness=1)

        return image

    def visualize_images_after_preprocessing(self, image, image_objects):

        print("The visualized image shape is: {}".format(image.shape))
        n_objects = 0
        for obj in image_objects:
            n_objects += 1
            box_to_print = {'xmin': int(obj["box"][0]), 'ymin': int(obj["box"][1]),
                            'xmax': int(obj["box"][2]), 'ymax': int(obj["box"][3]),
                            'name': obj["name"]}
            self.colored_rectangle_writer(image=image, obj=box_to_print)

        print("There are {} objects".format(n_objects))
        print("Press ESC to continue")

        scipy.misc.imshow(image)

    def visualize_img_boxes(self, image, image_objects):

        print("The visualized image shape is: {}".format(image.shape))
        n_objects = 0

        for obj in image_objects:
            n_objects += 1
            self.colored_rectangle_writer(image=image, obj=obj)

        print("There are {} objects".format(n_objects))
        print("Press ESC to continue")

        # scipy.misc.imshow(image)
        return image

    def draw_boxes_on_image(self, image, boxes_to_print):

        for box in boxes_to_print:

            if self.parameters.car_pov_inference_mode is True and self.object_distance_from_top(
                    box) < self.parameters.min_distance_from_top and self.object_area(box) > self.parameters.max_area_distant_obj:
                continue
            else:
                image = self.colored_rectangle_writer(image=image, box=box)

        return image

    def run_net_and_get_predictions(self):

        video_mode = self.parameters.video_mode
        test_image_path = self.parameters.test_image_path
        test_video_path = self.parameters.test_video_path
        save_video_to_file = self.parameters.save_video_to_file
        framerate = self.parameters.framerate

        if video_mode is False:

            files = os.listdir(test_image_path)
            for filename in files:
                full_image_path = test_image_path + filename
                log.info("Inference mode on image: {}".format(full_image_path))

                image, pure_cv2_image = self.preprocessor.read_image(image_path=full_image_path)

                boxes_to_print = self.prediction.network_output_pipeline(images=[image], pure_cv2_images=[pure_cv2_image])

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                # Normalize image to trick cv2.rectagle writer to think it's a colored image
                # so that it puts colored rectagles and not black ones
                # todo maybe fix this shitty thing:
                image = image / self.parameters.data_preprocessing_normalize
                output_image = self.draw_boxes_on_image(image=image, boxes_to_print=boxes_to_print[0])

                cv2.imshow('Network output', output_image)

                cv2.waitKey()

        else:
            cap = cv2.VideoCapture(test_video_path)
            frame_n = 0
            if save_video_to_file is True:
                save_video_dir = test_video_path + '_output'

                if not os.path.exists(save_video_dir):
                    os.makedirs(save_video_dir)

                fourcc = cv2.VideoWriter_fourcc(*'XVID')
                video = cv2.VideoWriter(save_video_dir + '/' + self.parameters.video_output_name + '.avi', fourcc,
                                        framerate, (1600, 800))
                time_start = time.time()
                while cap.isOpened():
                    frame_n += 1
                    ret, pure_cv2_image = cap.read()

                    if ret is not True:
                        break

                    # cv2 reads in BGR from the video, transform in RGB before giving it to network!
                    image = cv2.cvtColor(pure_cv2_image, cv2.COLOR_BGR2RGB)
                    boxes_to_print = self.prediction.network_output_pipeline(image=image, pure_cv2_image=pure_cv2_image)
                    # go back to GBR in order to visualize correctly with cv2
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # normalize the image or cv2 will draw only black rectangles smh
                    image = image / self.parameters.data_preprocessing_normalize
                    output_image = self.draw_boxes_on_image(image=image,
                                                            boxes_to_print=boxes_to_print)
                    # output_image = scipy.misc.imresize(image, (self.parameters.input_h, self.parameters.input_w))

                    if save_video_to_file is True:
                        output_image = output_image * self.parameters.data_preprocessing_normalize
                        video.write(np.uint8(output_image))
                time_end = time.time()
                fps = frame_n / (time_end - time_start)
                print("{} frames have been processed".format(frame_n))
                print("The fps were: {}".format(fps))
                video.release()


            else:
                # time_start = time.time()

                while cap.isOpened():
                    frame_n += 1
                    ret, pure_cv2_image = cap.read()

                    if ret is not True:
                        break

                    # cv2 reads in BGR from the video, transform in RGB before giving it to network!
                    image = cv2.cvtColor(pure_cv2_image, cv2.COLOR_BGR2RGB)
                    # Passing the image in a list because the function expects a batch of data
                    boxes_to_print = self.prediction.network_output_pipeline(images=[image], pure_cv2_images=[pure_cv2_image])
                    boxes_to_print = boxes_to_print[0]  # the return is for the whole batch
                    # go back to GBR in order to visualize correctly with cv2
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                    # normalize the image or cv2 will draw only black rectangles smh
                    image = image / self.parameters.data_preprocessing_normalize
                    output_image = self.draw_boxes_on_image(image=image, boxes_to_print=boxes_to_print)

                    name = 'image' + str(frame_n)
                    cv2.imshow('image', output_image)
                    # cv2.moveWindow('image', 0, 0)
                    key = cv2.waitKey()
                    if key == 27: break
                # time_end = time.time()
                # fps = frame_n / (time_end - time_start)
                # print("{} frames have been processed".format(frame_n))
                # print("The fps were: {}".format(fps))
