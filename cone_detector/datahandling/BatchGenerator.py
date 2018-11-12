from copy import deepcopy

import numpy as np
from visualization.Visualization import Visualization
import logging
log = logging.getLogger()
import math
class BatchGenerator(object):
    def __init__(self, parameters):
        self.parameters = parameters
        self.num_batches = None

    def get_generator(self, dataset, preprocessor, visualizer):

        n_images = len(dataset)

        self.num_batches = math.ceil(n_images / self.parameters.batch_size)

        shuffled_indices = np.random.permutation(np.arange(n_images))
        l_bound = 0
        r_bound = min(self.parameters.batch_size, n_images)

        n_classes = self.parameters.n_classes
        visualize_dataset = self.parameters.visualize_dataset
        visualize_preprocessed_images = self.parameters.visualize_preprocessed_images

        obj_class_probl = np.identity(n_classes, dtype=np.float32)

        while l_bound < n_images:
            batch_size = r_bound - l_bound
            batch_image_idx = 0

            x_batch_shape = (batch_size, self.parameters.input_h, self.parameters.input_w, self.parameters.input_depth)

            y_batch_shape = self.parameters.true_values_shape
            y_batch_shape[0] = batch_size

            x_batch = np.zeros(shape=x_batch_shape, dtype=np.float32)
            y_batch = np.zeros(shape=y_batch_shape, dtype=np.float32)
            filenames_batch = [None]*batch_size


            for index in shuffled_indices[l_bound:r_bound]:

                if visualize_dataset is True:
                    # log.info(dataset[index])
                    log.info("the visualized image is the output of the datset parsing, it has not yet been preprocessed")
                    visualizer.visualize_img_before_preprocessing(image_annotation=dataset[index])

                image, objects, filename = preprocessor.preprocess_for_training(dataset[index])
                filenames_batch[batch_image_idx] = filename

                if visualize_preprocessed_images is True:
                    log.info("The visualized image is the output of the preprocessing, input of the loss")
                    visualizer.visualize_images_after_preprocessing(image=image, image_objects=objects)


                # construct output from object's position and size
                for obj in objects:
                    x_grid = int(obj["x_grid"])
                    y_grid = int(obj["y_grid"])

                    obj_idx = self.parameters.labels_list.index(obj['name'])

                    # Note that the values are stored as float to avoid losing precision
                    # We dont need to replicate [box] and [1.0] for self.parameters.n_anchors because we are indexing it with :

                    y_batch[batch_image_idx, y_grid, x_grid, :, 0:4] = obj["box"]
                    y_batch[batch_image_idx, y_grid, x_grid, :, 4] = 1.0 #confidence
                    y_batch[batch_image_idx, y_grid, x_grid, :, 5:5 + n_classes] = obj_class_probl[obj_idx]
                    y_batch[batch_image_idx, y_grid, x_grid, :, 5 + n_classes] = obj["x_grid_rel"]
                    y_batch[batch_image_idx, y_grid, x_grid, :, 5 + n_classes + 1] = obj["y_grid_rel"]

                x_batch[batch_image_idx] = deepcopy(image)

                batch_image_idx += 1

            # Update batch boundaries
            l_bound = r_bound
            r_bound = min(r_bound + batch_size, n_images)

            # This function uses yield, when it's called from outside it will run the code untill the yield and return
            # the first value of the while loop in which the yield is contained
            # the while loop goes over the whole dataset, batch by batch
            # so every time the function is called it should return a new batch
            yield x_batch, y_batch, filenames_batch
