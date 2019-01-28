if __name__ == "__main__":
    pass
else:
    import logging
    from copy import deepcopy

    import numpy as np

    log = logging.getLogger()
    import math

class BatchGenerator(object):
    def __init__(self, parameters, dataset=None, preprocessor=None, visualizer=None, store_batch_y=True):
        self.parameters = parameters
        self.dataset = None
        self.preprocessor = None
        self.visualizer = None
        self.n_images = None
        self.num_batches = None
        self.stored_y_batch = None
        self.store_batch_y = store_batch_y
        self.store_batch_y_done = False
        self.y_batch_shape = None
        self.x_batch_shape = None
        self.stored_y_filenames = list()
        self.set_dataset(dataset, preprocessor, visualizer)


    def __embedded_visualizer(self, index):
        if self.parameters.visualize_dataset is True:
            # log.info(dataset[index])
            log.info("the visualized image is the output of the datset parsing, it has not yet been preprocessed")
            self.visualizer.visualize_img_before_preprocessing(image_annotation=self.dataset[index])

    def __embedded_proprocessed_visualizer(self, image, objects):
        if self.parameters.visualize_preprocessed_images is True:
            log.info("The visualized image is the output of the preprocessing, input of the loss")
            self.visualizer.visualize_images_after_preprocessing(image=image, image_objects=objects)

    def set_dataset(self, dataset=None, preprocessor=None, visualizer=None):
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.visualizer = visualizer
        if dataset is not None:
            self.n_images = len(dataset)

            self.num_batches = math.ceil(self.n_images / self.parameters.batch_size)
            log.info("Batch generator received a dataset of {} images - {} batches generated".format(self.n_images, self.num_batches))

            self.y_batch_shape = self.parameters.true_values_shape
            self.x_batch_shape = [None, self.parameters.input_h, self.parameters.input_w, self.parameters.input_depth]

            if self.store_batch_y is True:
                self.generate_stored_batch_y()

    def get_generator(self):
        if self.store_batch_y is True:
            yield from self.get_generator_stored_batch_y()
        else:
            yield from self.get_generator_no_prefetch()

    def generate_stored_batch_y(self):

        log.info("Executing batch_y prefetching")
        n_classes = self.parameters.n_classes
        obj_class_probl = np.identity(n_classes, dtype=np.float32)
        n_anchors = self.parameters.n_anchors
        stored_y_batch_shape = self.y_batch_shape
        stored_y_batch_shape[0] = self.n_images
        stored_y_batch = np.zeros(shape=stored_y_batch_shape, dtype=np.float32)

        for image_index, image in enumerate(self.dataset):
            objects, filename = self.preprocessor.preprocess_for_training(image, object_only=True)

            self.stored_y_filenames.append(filename)

            # construct output from object's position and size
            for obj in objects:
                x_grid = int(obj["x_grid"])
                y_grid = int(obj["y_grid"])

                obj_idx = self.parameters.labels_list.index(obj['name'])

                # Note that the values are stored as float to avoid losing precision
                # We dont need to replicate [box] and [1.0] for self.parameters.n_anchors because we are indexing it with :
                for box in range(n_anchors):
                    stored_y_batch[image_index, y_grid, x_grid, box, 0:4] = obj["box_oneb"]
                    stored_y_batch[image_index, y_grid, x_grid, box, 4] = 1.0  # confidence
                    stored_y_batch[image_index, y_grid, x_grid, box, 5:5 + n_classes] = obj_class_probl[obj_idx]
                    stored_y_batch[image_index, y_grid, x_grid, box, 5 + n_classes] = obj["x_grid_rel"]
                    stored_y_batch[image_index, y_grid, x_grid, box, 5 + n_classes + 1] = obj["y_grid_rel"]


        self.stored_y_batch = stored_y_batch
        self.store_batch_y_done = True
        log.info("Prefetching completed")

    def get_generator_stored_batch_y(self):
        shuffled_indices = np.random.permutation(np.arange(self.n_images))

        l_bound = 0
        r_bound = min(self.parameters.batch_size, self.n_images)

        while l_bound < self.n_images:
            batch_size = r_bound - l_bound

            self.x_batch_shape[0] = batch_size
            self.y_batch_shape[0] = batch_size

            x_batch = np.empty(shape=self.x_batch_shape, dtype=np.float32)
            y_batch = np.empty(shape=self.y_batch_shape, dtype=np.float32)

            filenames_batch = list()

            for in_batch_index, global_image_index in enumerate(shuffled_indices[l_bound:r_bound]):
                x_batch[in_batch_index], filename = self.preprocessor.preprocess_for_training(self.dataset[global_image_index], image_only=True)

                filenames_batch.append(filename)
                assert (self.stored_y_filenames[global_image_index] == filename), "Filename mismatch? {} {}".format(self.stored_y_filenames[
                                                                                                                        global_image_index], filename)
                y_batch[in_batch_index] = self.stored_y_batch[global_image_index]

            # Update batch boundaries
            l_bound = r_bound
            r_bound = min(r_bound + batch_size, self.n_images)
            yield x_batch, y_batch, filenames_batch

    def get_generator_no_prefetch(self):
        shuffled_indices = np.random.permutation(np.arange(self.n_images))

        l_bound = 0
        r_bound = min(self.parameters.batch_size, self.n_images)
        n_anchors = self.parameters.n_anchors
        n_classes = self.parameters.n_classes

        # diagonal matrix with 1 on the class
        obj_class_probl = np.identity(n_classes, dtype=np.float32)

        while l_bound < self.n_images:
            batch_size = r_bound - l_bound
            batch_image_idx = 0

            x_batch_shape = (batch_size, self.parameters.input_h, self.parameters.input_w, self.parameters.input_depth)

            y_batch_shape = self.parameters.true_values_shape
            y_batch_shape[0] = batch_size

            x_batch = np.empty(shape=x_batch_shape, dtype=np.float32)
            y_batch = np.zeros(shape=y_batch_shape, dtype=np.float32)
            filenames_batch = [None for _ in range(batch_size)]

            for index in shuffled_indices[l_bound:r_bound]:

                image, objects, filename = self.preprocessor.preprocess_for_training(self.dataset[index])
                filenames_batch[batch_image_idx] = filename

                self.__embedded_visualizer(index)
                self.__embedded_proprocessed_visualizer(image, objects)

                # construct output from object's position and size
                for obj in objects:
                    x_grid = int(obj["x_grid"])
                    y_grid = int(obj["y_grid"])

                    obj_idx = self.parameters.labels_list.index(obj['name'])

                    # Note that the values are stored as float to avoid losing precision
                    # We dont need to replicate [box] and [1.0] for self.parameters.n_anchors because we are indexing it with :
                    for box in range(n_anchors):
                        y_batch[batch_image_idx, y_grid, x_grid, box, 0:4] = obj["box_oneb"]
                        y_batch[batch_image_idx, y_grid, x_grid, box, 4] = 1.0  # confidence

                        y_batch[batch_image_idx, y_grid, x_grid, box, 5:5 + n_classes] = obj_class_probl[obj_idx]
                        y_batch[batch_image_idx, y_grid, x_grid, box, 5 + n_classes] = obj["x_grid_rel"]
                        y_batch[batch_image_idx, y_grid, x_grid, box, 5 + n_classes + 1] = obj["y_grid_rel"]

                x_batch[batch_image_idx] = deepcopy(image)

                batch_image_idx += 1

            # Update batch boundaries
            l_bound = r_bound
            r_bound = min(r_bound + batch_size, self.n_images)

            # This function uses yield, when it's called from outside it will run the code untill the yield and return
            # the first value of the while loop in which the yield is contained
            # the while loop goes over the whole dataset, batch by batch
            # so every time the function is called it should return a new batch
            yield x_batch, y_batch, filenames_batch
