import numpy as np
import os

np.set_printoptions(threshold='nan')


test_img_dir = '/home/nico/semester_project/cone_detector_data/validation/validation_images/'
fsg_output_dir = '/home/nico/semester_project/FSG_YOLO/pytorch-yolo2-marvis/outputs/'


# def get_yolo_fsg_outputs(self):
#     os.chdir('/home/nico/semester_project/FSG_YOLO/pytorch-yolo2-marvis/')
#
#     for image in os.listdir(test_img_dir):
#         image_path = test_img_dir + image
#         # print(image_path)
#         command = 'python2 detect.py cfg/yolo-voc-5-class-and-anchors.cfg 000400_25June2018_smallBoxes.weights ' + image_path
#         # print(command)
#         os.system(command)


def get_accuracy_fsg_output(self):

    for out in os.listdir(self.fsg_output_dir):
        output_path = self.fsg_output_dir + out
        output = np.load(output_path)
        network_output = np.reshape

        image_name = out[-10:-4] + 'jpg'
        image_path = test_img_dir + image_name
        image = self.preprocessor.read_image(image_path=image_path)
        output_boxes = self.prediction.get_output_boxes(netout=network_output)
        output_boxes = self.prediction.non_max_suppression(boxes=output_boxes)
        boxes_to_print = self.prediction.get_final_boxes(image=image,
                                              boxes=output_boxes)


get_accuracy_fsg_output()
# get_yolo_fsg_outputs(test_img_dir)