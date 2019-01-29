if __name__ == "__main__":
    pass
else:
    import logging

    import numpy as np

    log = logging.getLogger()

from datahandling.BatchGenerator import BatchGenerator
from visualization.BoundBox import BoundBox


class BatchGeneratorProteins(BatchGenerator):

    def generate_stored_batch_y(self):

        log.info("Executing batch_y prefetching")
        n_classes = self.parameters.n_classes
        obj_class_probl = np.identity(n_classes, dtype=np.float32)
        n_anchors = self.parameters.n_anchors
        stored_y_batch_shape = self.y_batch_shape
        stored_y_batch_shape[0] = self.n_images
        stored_y_batch = np.zeros(shape=stored_y_batch_shape, dtype=np.float32)
        self.anchor_assignements = np.zeros(shape=(n_anchors, ), dtype=np.int32)

        self.iou_sum = 0
        obj_counter = 0
        for image_index, image in enumerate(self.dataset):
            objects, filename = self.preprocessor.preprocess_for_training(image, object_only=True)

            self.stored_y_filenames.append(filename)

            # construct output from object's position and size
            for obj in objects:
                obj_counter = obj_counter + 1
                x_grid = int(round(obj["x_grid"]))
                y_grid = int(round(obj["y_grid"]))

                x_grid = min(x_grid, self.parameters.output_w - 1)
                y_grid = min(y_grid, self.parameters.output_h - 1)

                obj_idx = self.parameters.labels_list.index(obj['name'])

                confidences = self.get_confidences(obj, y_grid, x_grid)

                for box_idx in range(n_anchors):
                    stored_y_batch[image_index, y_grid, x_grid, box_idx, 0:4] = obj["box_oneb"]
                    stored_y_batch[image_index, y_grid, x_grid, box_idx, 4] = confidences[box_idx]
                    stored_y_batch[image_index, y_grid, x_grid, box_idx, 5:5 + n_classes] = obj_class_probl[obj_idx]
                    stored_y_batch[image_index, y_grid, x_grid, box_idx, 5 + n_classes] = obj["x_grid_rel"]
                    stored_y_batch[image_index, y_grid, x_grid, box_idx, 5 + n_classes + 1] = obj["y_grid_rel"]

        self.stored_y_batch = stored_y_batch
        self.store_batch_y_done = True
        iou_avg = self.iou_sum / obj_counter
        log.info("Prefetching completed - anchor assignment: {} - iou_avg {}".format(self.anchor_assignements, iou_avg))

    def get_confidences(self, obj, y_grid, x_grid):
        anchors = self.parameters.anchors

        best_iou = 0
        best_index = -1
        # print(obj)
        '''xmin = obj["xmin_oneb"] * self.parameters.output_w
        xmax = obj["xmax_oneb"] * self.parameters.output_w
        ymin = obj["ymin_oneb"] * self.parameters.output_h
        ymax = obj["ymax_oneb"] * self.parameters.output_h'''

        xmin = obj["xmin_grid"]
        xmax = obj["xmax_grid"]
        ymin = obj["ymin_grid"]
        ymax = obj["ymax_grid"]

        '''print("OBJ")
        print(xmin)
        print(xmax)
        print(ymin)
        print(ymax)
        print("----")'''

        obj_box = BoundBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, probs=None, class_type=None, conf=None)
        #print("self iou", obj_box.iou(obj_box))
        for anchor_idx, (anchor_w, anchor_h) in enumerate(zip(self.parameters.anchors_w, self.parameters.anchors_h)):

            xmin = x_grid - anchor_w
            xmax = x_grid + anchor_w
            ymin = y_grid - anchor_h
            ymax = y_grid + anchor_h

            anchor_box = BoundBox(xmin=xmin, xmax=xmax, ymin=ymin, ymax=ymax, probs=None, class_type=None, conf=None)

            iou = anchor_box.iou(obj_box)
            # print("anchor", anchor_idx)
            # print(anchor_w)
            # print(anchor_h)
            # print(xmin)
            # print(xmax)
            # print(ymin)
            # print(ymax)
            # print(iou)
            # print("*****-")
            if iou > best_iou:
                best_iou = iou
                best_index = anchor_idx

        confidences = np.zeros(shape=(len(anchors),))
        confidences[best_index] = 1
        self.anchor_assignements[best_index] = self.anchor_assignements[best_index] + 1
        self.iou_sum = self.iou_sum + best_iou
        # log.info("Found best IoU with anchor {} = {}".format(best_index, best_iou))

        return confidences

    def get_generator_no_prefetch(self):
        raise NotImplementedError
