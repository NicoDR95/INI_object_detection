# Step 1: divide images as classes

import tensorflow as tf
# from datahandling.object_detection.utils import dataset_util

import numpy as np


class TFRecordsConverter(object):

    def __init__(self, parameters, dataset, data_preprocessing, output_path):
        self.parameters = parameters
        self.dataset = dataset
        self.data_preprocessing = data_preprocessing
        self.output_path = output_path

    def bytes_feature(self, value):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    def convert_data_to_tf_example(self,
                                   height,        # Image height
                                   width,               # Image width
                                   channels,
                                   filename,            # Filename of the image. Empty if image is not from file
                                   encoded_image,       # Encoded image bytes
                                   image_format,        # b'jpeg' or b'png'
                                   objects,
                                   true_value):
        xmins = []  # List of normalized left x coordinates in bounding box (1 per box)
        xmaxs = []  # List of normalized right x coordinates in bounding box (1 per box)
        ymins = []  # List of normalized top y coordinates in bounding box (1 per box)
        ymaxs = []  # List of normalized bottom y coordinates in bounding box (1 per box)
        classes_text = []  # List of string class name of bounding box (1 per box)
        classes = []  # List of integer class id of bounding box (1 per box)

        for box in objects:
            # if box['occluded'] is False:
            # print("adding box")
            xmins.append(float(box['xmin']))
            xmaxs.append(float(box['xmax']))
            ymins.append(float(box['ymin']))
            ymaxs.append(float(box['ymax']))
            classes_text.append(box['name'].encode())
            # TODO STORE CLASS FROM NAME
            # classes.append(int(LABEL_DICT[box['label']]))

        # tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))
        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/encoded': self.bytes_feature(encoded_image),
            # 'image/height': self.int64_feature(height),
            # 'image/width': self.int64_feature(width),
            # 'image/channels': self.int64_feature(channels),
            # 'image/filename': self.bytes_feature(filename.encode()),
            # 'image/source_id': self.bytes_feature(filename.encode()),
            'image/true_value': self.bytes_feature(true_value)
            # 'image/object/bbox/difficult':
            # 'image/object/bbox/label':
            # 'image/object/bbox/label_text':
            # 'image/object/bbox/truncated':
            # 'image/shape':
            # # TODO FIX AND RESTORE
            # 'image/format': self.bytes_feature(image_format),
            # 'image/object/bbox/xmin': self.float_list_feature(xmins),
            # 'image/object/bbox/xmax': self.float_list_feature(xmaxs),
            # 'image/object/bbox/ymin': self.float_list_feature(ymins),
            # 'image/object/bbox/ymax': self.float_list_feature(ymaxs),
            # 'image/object/class/text': self.bytes_list_feature(classes_text),
            # 'image/object/class/label': self.int64_list_feature(classes),
        }))

        return tf_example

    def convert_dataset(self):

        n_classes = self.parameters.n_classes
        output_h = self.parameters.output_h
        output_w = self.parameters.output_w
        n_anchors = self.parameters.n_anchors
        obj_class_probl = np.identity(n_classes, dtype=np.float32)
        true_value_shape = [output_h, output_w, n_anchors, 4 + 1 + n_classes + 2]
        image_channels = 3

        writer = tf.python_io.TFRecordWriter(self.output_path + 'train_000.tfrecord')
        # writer = tf.python_io.TFRecordWriter('/media/nico/StorageBackup/tfrecord_output/' + 'train_000.tfrecord')
        # Read the dataset xmls
        all_image_anns = self.dataset.get_dataset_dict()

        for image_ann in all_image_anns:
            image, objects = self.data_preprocessing.preprocess_for_training(image_ann)
            true_value = np.zeros(shape=true_value_shape, dtype=np.float32)

            for obj in objects:
                x_grid = int(obj["x_grid"])
                y_grid = int(obj["y_grid"])

                obj_idx = self.parameters.labels_list.index(obj['name'])

                # Note that the values are stored as float to avoid losing precision
                # We dont need to replicate [box] and [1.0] for self.parameters.n_anchors because we are indexing it with :

                true_value[y_grid, x_grid, :, 0:4] = obj["box"]
                true_value[y_grid, x_grid, :, 4] = 1.0  # confidence
                true_value[y_grid, x_grid, :, 5:5 + n_classes] = obj_class_probl[obj_idx]
                true_value[y_grid, x_grid, :, 5 + n_classes] = obj["x_grid_rel"]
                true_value[y_grid, x_grid, :, 5 + n_classes + 1] = obj["y_grid_rel"]

            tf_example = self.convert_data_to_tf_example(
                height=image_ann["height"],             # Image height
                width=image_ann["width"],               # Image width
                channels= image_channels,
                filename=image_ann["filename"],         # Filename of the image. Empty if image is not from file
                encoded_image=image.tostring(),          # todo image_ann["height"],  # Encoded image bytes
                image_format="JPEG",                    # b'jpeg' or b'png'
                objects=image_ann["object"],            # List of normalized left x coordinates in bounding box (1 per box)
                true_value=true_value.tostring()         # true values array for the loss
            )

            writer.write(tf_example.SerializeToString())

        writer.close()


if __name__ == '_main_':
    tf.app.run()


# # Instantiate the batch generator
        # batches = generator.get_generator(dataset=all_image_anns,
        #                                              preprocessor=data_preprocessing,
        #                                              visualizer=visualizer)
        # # Iterate on the generator to make it process the whole dataset
        # for batch_iter_counter, (batch_images, batch_true_val) in enumerate(batches):
        #     image = np.squeeze(batch_images, axis=0)
        # for image_ann in all_image_anns:
        #
        # print(all_image_anns)
        # exit()