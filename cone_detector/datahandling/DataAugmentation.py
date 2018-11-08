# DONE: Data augmentation: => mirror, rotation (few degrees, 90 max), luminosity, gaussian noise,
# TODO: zoom in & zoom out, offset image up and down and left and and right (wrap of black padding),
# TODO: paste cones images on other images( also with wrong colors), images from VOC without cones (few of these)

import random
from random import shuffle
import numpy as np
import cv2
import imutils
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
from copy import deepcopy
import logging

log = logging.getLogger()


class DataAugmentation(object):

    def __init__(self, parameters,
                 image_dir,
                 annotations_dir,
                 augmented_image_dir,
                 augmented_annotations_dir,
                 dataset, visualization,
                 check_augmentation,
                 crop_for_aspect_ratio,
                 target_aspect_ratio,
                 shuffle_functions,
                 mirror,
                 translate,
                 translate_x,
                 translate_y,
                 rotate,
                 rotate_range,
                 rotate_no_cut,
                 illumination,
                 gamma_change_min,
                 gamma_change_max,
                 gaussian_noise,
                 gaussian_mean,
                 gaussian_standard_dev_min,
                 gaussian_standard_dev_max,
                 salt_pepper_noise,
                 salt_vs_pepper_min,
                 salt_vs_pepper_max,
                 salt_pepper_amount_min,
                 salt_pepper_amount_max,
                 poisson_noise,
                 speckle_noise,
                 gaussian_blur,
                 gaussian_kernel_min,
                 gaussian_kernel_max,
                 gaussian_sigma,
                 average_blur,
                 average_kernel_min,
                 average_kernel_max
                 ):
        self.parameters = parameters
        self.image_dir = image_dir
        self.annotations_dir = annotations_dir
        self.augmented_image_dir = augmented_image_dir
        self.augmented_annotations_dir = augmented_annotations_dir
        self.dataset = dataset
        self.visualization = visualization
        self.check_augmentation = check_augmentation
        self.shuffle_functions = shuffle_functions
        self.func_list = []
        self.translate_x = translate_x
        self.translate_y = translate_y
        self.rotate_range = rotate_range
        self.rotate_no_cut = rotate_no_cut
        self.gamma_change_min = gamma_change_min
        self.gamma_change_max = gamma_change_max
        self.gaussian_mean = gaussian_mean
        self.gaussian_standard_dev_min = gaussian_standard_dev_min
        self.gaussian_standard_dev_max = gaussian_standard_dev_max
        self.salt_vs_pepper_min = salt_vs_pepper_min
        self.salt_vs_pepper_max = salt_vs_pepper_max
        self.salt_pepper_amount_min = salt_pepper_amount_min
        self.salt_pepper_amount_max = salt_pepper_amount_max
        self.gaussian_kernel_min = gaussian_kernel_min
        self.gaussian_kernel_max = gaussian_kernel_max
        self.gaussian_sigma = gaussian_sigma
        self.average_kernel_min = average_kernel_min
        self.average_kernel_max = average_kernel_max
        self.crop_for_aspect_ratio = crop_for_aspect_ratio
        self.target_aspect_ratio = target_aspect_ratio
        if mirror is True:
            self.func_list.append(self.mirror)
        if translate is True:
            self.func_list.append(self.translate)
        if rotate is True:
            self.func_list.append(self.rotate)
        if illumination is True:
            self.func_list.append(self.change_illumination)
        if gaussian_noise is True:
            self.func_list.append(self.add_gaussian_noise)
        if salt_pepper_noise is True:
            self.func_list.append(self.add_salt_pepper_noise)
        if poisson_noise is True:
            self.func_list.append(self.add_poisson_noise)
        if speckle_noise is True:
            self.func_list.append(self.add_speckle_noise)
        if gaussian_blur is True:
            self.func_list.append(self.add_gaussian_blur)
        if average_blur is True:
            self.func_list.append(self.add_average_blur)

    # We read the imge with a normal cv2 read, for visualization we use the same function used by the inference
    # to be sure everything is allright
    def read_image_cv2(self, image_ann):
        path = image_ann['filename']
        image_obj = image_ann['object']
        image = cv2.imread(self.image_dir + path)
        h, w, c = image.shape

        return image, image_obj, h, w, c

    # ~~~~~~~~~~~~~~~~~~~~~~ Flipping ~~~~~~~~~~~~~~~~~~~~~~
    def mirror(self, image, image_obj, h, w, c):
        output_image = cv2.flip(image, 1)  # Flip the image around y-axis (1)

        for obj in image_obj:
            xmin = obj['xmin']
            obj['xmin'] = w - obj['xmax']
            obj['xmax'] = w - xmin
        return output_image, image_obj, h, w, c

    # ~~~~~~~~~~~~~~~~~~~~~~ Translation~~~~~~~~~~~~~~~~~~~~~~
    def translate(self, image, image_obj, h, w, c):
        translate_x = np.random.randint(- self.translate_x, self.translate_x + 1)
        translate_y = np.random.randint(- self.translate_y, self.translate_y + 1)
        translation_matrix = np.float32([[1, 0, translate_y], [0, 1, translate_x]])
        output_image = cv2.warpAffine(image, translation_matrix, (w, h))

        # Fix the BB
        original_obj = deepcopy(image_obj)
        for obj in image_obj:
            width = obj['xmax'] - obj['xmin']
            height = obj['ymax'] - obj['ymin']
            center_x = int(0.5 * (obj['xmin'] + obj['xmax']))
            center_y = int(0.5 * (obj['ymin'] + obj['ymax']))
            translated_center_x = center_x + translation_matrix[0][2]
            translated_center_y = center_y + translation_matrix[1][2]
            obj['xmin'] = int(translated_center_x - width / 2)
            obj['xmax'] = int(translated_center_x + width / 2)
            obj['ymin'] = int(translated_center_y - height / 2)
            obj['ymax'] = int(translated_center_y + height / 2)

            # if an object happens to be out of the image (or on the edge) undo the translation
            if obj['xmin'] < 0 or obj['xmax'] > w or obj['ymin'] < 0 or obj['ymax'] > h:
                output_image = image
                image_obj = original_obj
                break
        h, w, c = output_image.shape

        return output_image, image_obj, h, w, c

    # ~~~~~~~~~~~~~~~~~~~~~~ Rotation ~~~~~~~~~~~~~~~~~~~~~~
    def rotate(self, image, image_obj, h, w, c):
        angle = np.random.randint(-self.rotate_range, self.rotate_range + 1)
        if self.rotate_no_cut is True:
            out_image, transformation_matrix = self.rotate_no_cutting(image=image, angle=angle)
        elif self.rotate_no_cut is False:
            out_image, transformation_matrix = self.rotate_cut_angles(image=image, angle=angle)
        else:
            log.error("Please specify a valid value for rotate_cut in main. Either True or False")
            exit()
        # Fix the BB
        for obj in image_obj:
            width = obj['xmax'] - obj['xmin']
            height = obj['ymax'] - obj['ymin']
            center_x = int(0.5 * (obj['xmin'] + obj['xmax']))
            center_y = int(0.5 * (obj['ymin'] + obj['ymax']))
            # fake_image = np.zeros((h, w, c))
            # fake_image[center_y][center_x][:] = 255

            # rotated_fake_image, _ = self.rotate_no_cutting(image=image, angle=angle)

            center = np.array(([center_x],
                               [center_y]))
            rotation_matrix = np.zeros((2, 2))
            rotation_matrix[0][0] = transformation_matrix[0][0]
            rotation_matrix[0][1] = transformation_matrix[0][1]
            rotation_matrix[1][0] = transformation_matrix[1][0]
            rotation_matrix[1][1] = transformation_matrix[1][1]

            rotated_center = np.matmul(rotation_matrix, center)
            transformed_center_x = rotated_center[0] + transformation_matrix[0][2]
            transformed_center_y = rotated_center[1] + transformation_matrix[1][2]

            obj['xmin'] = int(transformed_center_x - width / 2)
            obj['xmax'] = int(transformed_center_x + width /2)
            obj['ymin'] = int(transformed_center_y - height / 2)
            obj['ymax'] = int(transformed_center_y + height / 2)

        h, w, c = out_image.shape

        return out_image, image_obj, h, w, c

    # copy pasted function imutils.rotate_bounds, modified to return also the matrix:
    def rotate_cut_angles(self, image, angle, center=None, scale=1.0):
        # grab the dimensions of the image
        (h, w) = image.shape[:2]
        # if the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2)
        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale)
        out_image = cv2.warpAffine(image, M, (w, h))
        # return the rotated image
        return out_image, M

    # copy pasted function imutils.rotate, modified to return also the matrix:
    def rotate_no_cutting(self, image, angle):
        # grab the dimensions of the image and then determine the
        # center
        (h, w) = image.shape[:2]
        (cX, cY) = (w / 2, h / 2)
        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        out_image = cv2.warpAffine(image, M, (nW, nH))

        return out_image, M

    # ~~~~~~~~~~~~~~~~~~~~~~ Illumination ~~~~~~~~~~~~~~~~~~~~~~
    def change_illumination(self, image, image_obj, h, w, c):
        gamma = round(random.uniform(self.gamma_change_min, self.gamma_change_max), 2)
        out_image = self.adjust_gamma(image=image, gamma=gamma)

        return out_image, image_obj, h, w, c

    def adjust_gamma(self, image, gamma=1.0):
        # build a lookup table mapping the pixel values [0, 255] to
        # their adjusted gamma values
        invGamma = 1.0 / gamma
        table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(0, 256)]).astype("uint8")

        # apply gamma correction using the lookup table
        output_image = cv2.LUT(image, table)

        return output_image
    # ~~~~~~~~~~~~~~~~~~~~~~ Noise ~~~~~~~~~~~~~~~~~~~~~~

    def add_gaussian_noise(self, image, image_obj, h, w, c):
        mean = self.gaussian_mean
        standard_dev = np.random.randint(self.gaussian_standard_dev_min, self.gaussian_standard_dev_max + 1)
        gauss = np.random.normal(mean, standard_dev, (h, w, c))
        gauss = gauss.reshape(h, w, c)
        output_image = image + gauss
        return output_image, image_obj, h, w, c

    def add_salt_pepper_noise(self, image, image_obj, h, w, c):
        s_vs_p = random.uniform(self.salt_vs_pepper_min, self.salt_vs_pepper_max)
        amount = random.uniform(self.salt_pepper_amount_min, self.salt_pepper_amount_max)
        output_image = np.copy(image)
        # Salt mode
        num_salt = np.ceil(amount * image.size * s_vs_p)

        coords = [np.random.randint(0, i, int(num_salt)) for i in image.shape]
        output_image[coords] = 254
        # Pepper mode
        num_pepper = np.ceil(amount * image.size * (1. - s_vs_p))
        coords = [np.random.randint(0, i, int(num_pepper)) for i in image.shape]
        output_image[coords] = 0
        return output_image, image_obj, h, w, c

    def add_poisson_noise(self, image, image_obj, h, w, c):
        vals = len(np.unique(image))
        vals = 2 ** np.ceil(np.log2(vals))
        output_image = np.random.poisson(image * vals) / float(vals)
        return output_image, image_obj, h, w, c

    def add_speckle_noise(self, image, image_obj, h, w, c):
        gauss = np.random.randn(h, w, c)
        gauss = gauss.reshape(h, w, c)
        output_image = image + image * gauss
        return output_image, image_obj, h, w, c

    # ~~~~~~~~~~~~~~~~~~~~~~ Blur ~~~~~~~~~~~~~~~~~~~~~~
    def add_gaussian_blur(self, image, image_obj, h, w, c):
        gaussian_kernel = random.choice(range(self.gaussian_kernel_min, self.gaussian_kernel_max, 2))
        output_image = cv2.GaussianBlur(image, (gaussian_kernel, gaussian_kernel), self.gaussian_sigma)
        return output_image, image_obj, h, w, c

    def add_average_blur(self,image, image_obj, h, w, c):
        average_kernel = random.choice(range(self.average_kernel_min, self.average_kernel_max, 2))
        output_image = cv2.blur(image, (average_kernel, average_kernel), self.gaussian_sigma)
        return output_image, image_obj, h, w, c

    # ~~~~~~~~~~~~~~~~~~~~~~ aspect ratio cropper  ~~~~~~~~~~~~~~~~~~~~~~
    def aspect_ratio_cropper(self, image, image_obj, h, w, c):

        target_ratio = self.target_aspect_ratio

        if w//c == target_ratio:
            return image, image_obj, h, w, c
        else:
            target_h = w//2
            w_h_difference = w - h
            target_h_to_cut = target_h - w_h_difference

            to_cut_from_top = target_h_to_cut//2
            to_cut_from_bottom = to_cut_from_top
            ymin_list = []
            ymax_list = []
            for obj in image_obj:
                ymin_list.append(obj['ymin'])
                ymax_list.append(obj['ymax'])
            # y coordinates are referenced to top of image
            try:
                smallest_obj_coor = min(ymin_list)
                biggest_obj_coor = max(ymax_list)
            except ValueError:
                # if the image is empty cut it with no problems
                cropped_image = image[to_cut_from_top: h - to_cut_from_bottom, :]
                h, w, c = cropped_image.shape
                return cropped_image, image_obj, h, w, c

            if smallest_obj_coor >= to_cut_from_top and biggest_obj_coor <= h - to_cut_from_bottom:
                # then you can cut perfectly for the aspect ratio
                cropped_image = image[to_cut_from_top : h-to_cut_from_bottom, :]
                # adjust boxes (no need to move obj for cut from bottom)
                for obj in image_obj:
                    obj['ymin'] = obj['ymin'] - to_cut_from_top
                    obj['ymax'] = obj['ymax'] - to_cut_from_top
                h, w, c = cropped_image.shape
                return cropped_image, image_obj, h, w, c

            elif smallest_obj_coor < to_cut_from_top and biggest_obj_coor <= h - to_cut_from_bottom:
                error_from_top = to_cut_from_top - smallest_obj_coor
                to_cut_from_top = smallest_obj_coor
                # try to cut more from the bottom then:
                to_cut_from_bottom = to_cut_from_bottom + error_from_top + 1
                while True:
                    to_cut_from_bottom = to_cut_from_bottom - 1
                    if biggest_obj_coor > h - to_cut_from_bottom:       # if it's wrong make it smaller
                        continue
                    else:
                        break
                cropped_image = image[to_cut_from_top: h - to_cut_from_bottom, :]
                for obj in image_obj:
                    obj['ymin'] = obj['ymin'] - to_cut_from_top
                    obj['ymax'] = obj['ymax'] - to_cut_from_top
                h, w, c = cropped_image.shape
                return cropped_image, image_obj, h, w, c

            elif smallest_obj_coor >= to_cut_from_top and biggest_obj_coor > h - to_cut_from_bottom:
                error_from_bottom = biggest_obj_coor - (h - to_cut_from_bottom)
                to_cut_from_bottom = h - biggest_obj_coor
                # try to cut more from top then:
                to_cut_from_top = to_cut_from_top + error_from_bottom + 1
                while True:
                    to_cut_from_top = to_cut_from_top - 1
                    if smallest_obj_coor < to_cut_from_top:       # if it's wrong make it smaller
                        continue
                    else:
                        break
                cropped_image = image[to_cut_from_top: h - to_cut_from_bottom, :]
                for obj in image_obj:
                    obj['ymin'] = obj['ymin'] - to_cut_from_top
                    obj['ymax'] = obj['ymax'] - to_cut_from_top
                h, w, c = cropped_image.shape
                return cropped_image, image_obj, h, w, c

            elif smallest_obj_coor < to_cut_from_top and biggest_obj_coor > h - to_cut_from_bottom:
                # if both condition are wrong, just cut as much as you can
                to_cut_from_top = smallest_obj_coor
                to_cut_from_bottom = h - biggest_obj_coor
                cropped_image = image[to_cut_from_top: h - to_cut_from_bottom, :]
                for obj in image_obj:
                    obj['ymin'] = obj['ymin'] - to_cut_from_top
                    obj['ymax'] = obj['ymax'] - to_cut_from_top
                h, w, c = cropped_image.shape
                return cropped_image, image_obj, h, w, c

    # ~~~~~~~~~~~~~~~~~~~~~~ xml writer ~~~~~~~~~~~~~~~~~~~~~~
    def xml_file_writer(self, image_obj, image_name, w, h, c):
        image_obj_xml = []
        image_obj_xml.append({'filename': image_name,
                              'size': {'width': w, 'height': h, 'depth': c},
                              'object': []})

        for obj in image_obj:
            image_obj_xml.append({'object':{'name': obj['name'],
                                               'bndbox': {'xmin': obj['xmin'], 'ymin': obj['ymin'],
                                                          'xmax': obj['xmax'], 'ymax': obj['ymax']}}})
        xml = dicttoxml(image_obj_xml, attr_type=False, custom_root='annotation', item_func=lambda x: None)
        # The xml writer will put subsections for item with the tag Nonce, we remove them so we have nice xml
        xml = xml.replace(b'<None>', b'')
        xml = xml.replace(b'</None>', b'')
        xml = parseString(xml)
        xml = xml.toprettyxml()

        with open(self.augmented_annotations_dir + image_name[:-3] + 'xml', 'w+') as xml_file:
            xml_file.write(xml)
            xml_file.close()

    def visualize_augmented_dataset(self):
        dataset_annotations = self.dataset.get_dataset_dict(self.augmented_annotations_dir)
        print(dataset_annotations)
        print(len(dataset_annotations))

        for index in range(len(dataset_annotations)):
            print(index)
            print(dataset_annotations[index])
            self.visualization.visualize_img_before_preprocessing(image_annotation=dataset_annotations[index])

    def data_aug_pipeline(self):
        augmentation_run = self.parameters.augmentation_run
        dataset_anns = self.dataset.get_dataset_dict(self.annotations_dir)
        if self.shuffle_functions is True:
            shuffle(self.func_list)
        iter = 0
        for image_ann in dataset_anns:
            iter += 1
            image, image_obj, h, w, c = self.read_image_cv2(image_ann)
            # print(image_obj
            n_aug_func = random.randint(1, len(self.func_list)+1)
            for aug_func in self.func_list[0:n_aug_func]:

                image, image_obj, h, w, c = aug_func(image, image_obj, h, w, c)
                # if aug_image is None and aug_image_obj is None:
                #     continue

                if self.check_augmentation is True:
                    image_to_check = self.visualization.draw_boxes_on_image(image, image_obj)
                    cv2.imshow('image', image_to_check)
                    cv2.waitKey()

            # crop the images to get closer to desired aspect ratio
            if self.crop_for_aspect_ratio is True:
                h, w, c = image.shape
                image, image_obj, h, w, c = self.aspect_ratio_cropper(image, image_obj, h, w, c)

            image_name = 'aug' + '_' + augmentation_run + '_' + image_ann['filename']
            cv2.imwrite(self.augmented_image_dir + image_name, image)

            self.xml_file_writer(image_obj=image_obj, image_name=image_name, w=w, h=h, c=c)

                # if iter == 50:
                #     exit()

