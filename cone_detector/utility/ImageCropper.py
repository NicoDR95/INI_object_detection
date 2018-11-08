import cv2
import os
import xml.etree.ElementTree as ET
from collections import OrderedDict
from dicttoxml import dicttoxml
from xml.dom.minidom import parseString
import logging
log = logging.getLogger()



crop_images_mode = True
adapt_annotations_mode = False

# image_dir = '/home/nico/semester_project/cone_detector_data/dataset/cones_dataset2018/car_pov_uncropped/images/'
image_dir = '/home/nico/semester_project/cone_detector_data/validation/validation_images_tocrop/'
# image_dest_dir = '/home/nico/semester_project/cone_detector_data/dataset/cones_dataset2018/car_pov_cropped/images/'
image_dest_dir = '/home/nico/semester_project/cone_detector_data/validation/validation_images_cropped/'
# annotations_dir = '/home/nico/semester_project/cone_detector_data/dataset/cones_dataset2018/car_pov_uncropped/annotations/'
annotations_dir = '/home/nico/semester_project/cone_detector_data/validation/validation_annotations_tocrop/'
# annotations_dest_dir = '/home/nico/semester_project/cone_detector_data/dataset/cones_dataset2018/car_pov_cropped/annotations/'
annotations_dest_dir = '/home/nico/semester_project/cone_detector_data/validation/validation_annotations_cropped/'

def crop_image(image_path):
    img = cv2.imread(image_path)
    crop_img = img[400:1200, :]
    # cv2.imshow("cropped", crop_img)
    # cv2.waitKey()
    return crop_img


if crop_images_mode is True:
    for image_name in os.listdir(image_dir):
        image_path = image_dir + image_name
        cropped = crop_image(image_path)
        cv2.imwrite(image_dest_dir + image_name, cropped)



def read_dataset(annotations_dir):
    img_anns = []
    classes = OrderedDict()
    for ann in os.listdir(annotations_dir):
        img = {'object': []}

        tree = ET.parse(annotations_dir + ann)

        for elem in tree.iter():
            if 'filename' in elem.tag:
                img_anns += [img]
                img['filename'] = elem.text
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
                                obj['xmin'] = int(round(float(dim.text)))
                            if 'ymin' in dim.tag:
                                obj['ymin'] = int(round(float(dim.text)))
                            if 'xmax' in dim.tag:
                                obj['xmax'] = int(round(float(dim.text)))
                            if 'ymax' in dim.tag:
                                obj['ymax'] = int(round(float(dim.text)))


    log.info("Dataset read finished. There are {} classes in this dataset".format(len(classes)))

    for class_key in classes.keys():
        log.info("Class {} has {} occurrencies".format(class_key, classes[class_key]))
    return img_anns


def object_adapter(image_obj):

    for obj in image_obj:
        obj['ymin'] = obj['ymin'] - 400
        obj['ymax'] = obj['ymax'] - 400

    return image_obj


def xml_file_writer(image_obj, image_name, w=1600, h=800, c=3):
    image_obj_xml = []
    image_obj_xml.append({'filename': image_name,
                          'size': {'width': w, 'height': h, 'depth': c},
                          'object': []})

    for obj in image_obj:
        image_obj_xml.append({'object': {'name': obj['name'],
                                         'bndbox': {'xmin': obj['xmin'], 'ymin': obj['ymin'],
                                                    'xmax': obj['xmax'], 'ymax': obj['ymax']}}})
    xml = dicttoxml(image_obj_xml, attr_type=False, custom_root='annotation', item_func=lambda x: None)
    # The xml writer will put subsections for item with the tag Nonce, we remove them so we have nice xml
    xml = xml.replace(b'<None>', b'')
    xml = xml.replace(b'</None>', b'')
    xml = parseString(xml)
    xml = xml.toprettyxml()

    with open(annotations_dest_dir + image_name[:-3] + 'xml', 'w+') as xml_file:
        xml_file.write(xml)
        xml_file.close()


if adapt_annotations_mode is True:

    image_annotations = read_dataset(annotations_dir=annotations_dir)

    for image_ann in image_annotations:

        image_name = image_ann['filename']
        image_obj = image_ann['object']
        image_obj = object_adapter(image_obj)
        xml_file_writer(image_obj=image_obj, image_name=image_name)


# cropped = crop_image( '/home/nico/semester_project/cone_detector_data/dataset/cones_dataset2018/to_crop/400378.jpg')
# cv2.imwrite(image_dir + '400378.jpg', cropped)