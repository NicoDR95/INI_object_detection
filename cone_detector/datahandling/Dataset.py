import logging
import xml.etree.ElementTree as ET
from collections import OrderedDict

import os

log = logging.getLogger()


class Dataset(object):

    def __init__(self, parameters, base_path, annotations_filelist):
        self.parameters = parameters
        self.base_path = base_path
        self.annotations_filelist = annotations_filelist

    def get_dataset_dict(self):

        img_anns = []
        classes = OrderedDict()


        if self.annotations_filelist is None:
            log.info("Reading dataset in dir {}".format(self.base_path))
            try:
                filelist = [ann for ann in os.listdir(self.base_path)]
            except FileNotFoundError:
                log.warning("The folder: {} does not exists => skipping this dataset folder".format(self.base_path))
                return None
        else:
            log.info("Reading dataset from filelist {}".format(self.annotations_filelist))
            with open(self.annotations_filelist) as f:
                content = f.readlines()

            # remove whitespace characters like `\n` at the end of each line and append xml
            filelist = ["/" + x.strip() + ".xml" for x in content]



        log.info("Found {} files".format(len(filelist)))

        for ann in filelist:
            img = {'object': []}
            file_to_parse = self.base_path + "/" + ann


            #print("file_to_parse", file_to_parse)
            tree = ET.parse(file_to_parse)

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
                            # if obj_name in self.parameters.labels_dict:
                            #    obj['class'] = self.parameters.labels_dict[obj['name']]
                            img['object'] += [obj]

                        if 'bndbox' in attr.tag:
                            for dim in list(attr):
                                if 'xmin' in dim.tag:
                                    obj['xmin'] = int(dim.text)
                                if 'ymin' in dim.tag:
                                    obj['ymin'] = int(dim.text)
                                if 'xmax' in dim.tag:
                                    obj['xmax'] = int(dim.text)
                                if 'ymax' in dim.tag:
                                    obj['ymax'] = int(dim.text)


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
