import xml.etree.ElementTree as ET
from collections import OrderedDict
import os
import logging
log = logging.getLogger()


class Dataset(object):

    def __init__(self, parameters, annotations_dir):
        self.parameters = parameters
        self.annotations_dir = annotations_dir

    def get_dataset_dict(self, particular_dir=None):
        log.info("Reading a dataset")
        img_anns = []
        classes = OrderedDict()

        if particular_dir is not None:
            self.annotations_dir = particular_dir
        try:
            for ann in os.listdir(self.annotations_dir):
                img = {'object': []}

                tree = ET.parse(self.annotations_dir + ann)

                for elem in tree.iter():
                    if 'filename' in elem.tag:
                        img_anns += [img]
                        img['filename'] = elem.text
                        # img['image_data'] = self.read_image_from_file(img['filename'])
                    if 'width' in elem.tag:
                        img['width'] = int(elem.text)
                    if 'height' in elem.tag:
                        img['height'] = int(elem.text)
                    if 'object' in elem.tag or 'part' in elem.tag:
                        obj = {}

                        for attr in list(elem):
                            if 'name' in attr.tag:
                                obj_name = attr.text
                                obj['name'] = obj_name
                                try:
                                    classes[obj_name] = classes[obj_name] + 1
                                except KeyError:
                                    log.info("New class found in dataset: {}".format(obj_name))
                                    classes[obj_name] = 1

                                # add additional label if class label available
                                #if obj_name in self.parameters.labels_dict:
                                #    obj['class'] = self.parameters.labels_dict[obj['name']]
                                img['object'] += [obj]

                            if 'bndbox' in attr.tag:
                                for dim in list(attr):
                                    if 'xmin' in dim.tag:
                                        obj['xmin'] = int(round(float(dim.text)))
                                    if 'ymin' in dim.tag:
                                        obj['ymin'] = int(round(float(dim.text)))
                                    if 'xmax' in dim.tag:
                                        obj['xmax'] = int(round(float(dim.text)))
                                    if 'ymax' in dim.tag:
                                        obj['ymax'] = int(round(float(dim.text)))
        except FileNotFoundError:
            log.info("The folder: {} does not exists => skipping this dataset folder".format(self.annotations_dir))
            return None

        for dataset_class in classes:
            if dataset_class not in self.parameters.labels_list:
                raise ValueError("Class {} found in dataset doesn't have a mach in labels_list.".format(dataset_class))

        if len(classes) != len(self.parameters.labels_list):
            log.warn("Len of found classes in dataset {} different from provided labels_dict {}".format(len(classes),
                                                                                                              len(self.parameters.labels_list)))

        log.info("Dataset read finished. There are {} classes in this dataset".format(len(classes)))

        for class_key in classes.keys():
            log.info("Class {} has {} occurrencies".format(class_key, classes[class_key]))

        return img_anns

    # def read_image_from_file(self, img_filename):
    #     image_path = self.images_dir + img_filename
    #     image = scipy.ndimage.imread(image_path, mode='RGB')
    #     image = np.array(image, dtype=np.float32)
    #     return image
