from operator import itemgetter

import numpy as np


class BoundBox(tuple):
    # Makes the class imumutable after creation
    __slots__ = []

    def __new__(cls, x, y, w, h, probs, conf, maxmin_x_rescale, maxmin_y_rescale, class_type, groundtruth, xmin, xmax, ymin, ymax):
        if groundtruth is False:
            area = w * h

            half_w = w / 2
            half_h = h / 2
            width_for_inters = [x - half_w, x + half_w]
            height_for_inters = [y - half_h, y + half_h]

            xmin = int((x - half_w) * maxmin_x_rescale)
            xmax = int((x + half_w) * maxmin_x_rescale)
            ymin = int((y - half_h) * maxmin_y_rescale)
            ymax = int((y + half_h) * maxmin_y_rescale)
            max_prob = np.amax(probs)
        else:
            area = None
            half_w = None
            half_h = None
            width_for_inters = None
            height_for_inters = None
            max_prob = None

        xmax_xmin_area = (ymax - ymin) * (xmax - xmin)

        assert (xmax_xmin_area >= 0)

        return tuple.__new__(cls,
                             (x, y, w, h, area, width_for_inters, height_for_inters, probs, conf, class_type, xmin, xmax, ymin,
                              ymax, max_prob, half_w, half_h, xmax_xmin_area))

    x = property(itemgetter(0))
    y = property(itemgetter(1))
    w = property(itemgetter(2))
    h = property(itemgetter(3))
    area = property(itemgetter(4))
    width_for_inters = property(itemgetter(5))
    height_for_inters = property(itemgetter(6))
    probs = property(itemgetter(7))
    conf = property(itemgetter(8))
    class_type = property(itemgetter(9))
    xmin = property(itemgetter(10))
    xmax = property(itemgetter(11))
    ymin = property(itemgetter(12))
    ymax = property(itemgetter(13))
    max_prob = property(itemgetter(14))
    half_w = property(itemgetter(15))
    half_h = property(itemgetter(16))
    xmax_xmin_area = property(itemgetter(17))

    def __area(self):
        return self.w * self.h

    def iou(self, box, accuracy_mode=False):
        intersection = self.intersect(box, accuracy_mode)

        if accuracy_mode is False:
            union = self.area + box.area - intersection
            iou = intersection / union
            return iou
        elif accuracy_mode is True:
            if self.class_type == box.class_type:
                pred_area = self.xmax_xmin_area
                true_area = box.xmax_xmin_area
                union = pred_area + true_area - intersection

                assert (union >= 0)

                return intersection / union  # iou
            else:
                return 0

    def intersect(self, box, accuracy_mode):
        if accuracy_mode is False:
            width_inters = self.__overlap(self.width_for_inters, box.width_for_inters)
            height_inters = self.__overlap(self.height_for_inters, box.height_for_inters)
            intersect_area = width_inters * height_inters
            return intersect_area

        elif accuracy_mode is True:
            width_inters = self.__overlap([self.xmin, self.xmax], [box.xmin, box.xmax])
            height_inters = self.__overlap([self.ymin, self.ymax], [box.ymin, box.ymax])
            intersect_area = width_inters * height_inters
            return intersect_area

    def __overlap(self, interval_a, interval_b):
        # calculate the overlap between two boxes
        # inputs: xmin and xmax (or ymin and ymax) coordinates of boxes
        # output: length of x (or y) segment of overlap

        x1, x2 = interval_a
        x3, x4 = interval_b
        if x3 < x1:
            if x4 < x1:
                return 0
            else:
                return min(x2, x4) - x1
        else:
            if x2 < x3:
                return 0
            else:
                return min(x2, x4) - x3

    def is_matrioska(self, box):
        # Check if the passed box is contained into the self box
        # Remember: the 0,0 point is the top_left corner

        x2 = self.x + self.half_w
        x4 = box.x + box.half_w

        if x2 > x4:
            x1 = self.x - self.half_w
            x3 = box.x - box.half_w

            if x1 < x3:
                y2 = self.y + self.half_h
                y4 = box.y + box.half_h

                if y2 > y4:
                    y1 = self.y - self.half_h
                    y3 = box.y - box.half_h

                    if y1 < y3:
                        return True

        return False
