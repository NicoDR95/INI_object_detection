import numpy as np

class BoundBox(object):

    def __init__(self, class_num, accuracy_mode):
        self.x = 0.
        self.y = 0.
        self.w = 0.
        self.h = 0.
        self.xmin = 0.
        self.xmax = 0.
        self.ymin = 0.
        self.ymax = 0.
        self.conf = 0.
        self.class_probs = np.zeros(class_num)
        self.accuracy_mode = accuracy_mode
        self.class_type = ' '

    def area(self):
        return self.w*self.h
    def iou(self, box, accuracy_mode=False):
        intersection = self.intersect(box, accuracy_mode)

        if accuracy_mode is False:
            union = self.w*self.h + box.w*box.h - intersection
            iou = intersection / union
            return iou
        elif accuracy_mode is True:
            pred_area = (self.ymax - self.ymin) * (self.xmax - self.xmin)
            true_area = (box.ymax - box.ymin) * (box.xmax - box.xmin)
            assert(pred_area >= 0)
            assert(true_area >= 0)
            union = pred_area + true_area - intersection
            assert(union >= 0)
            if self.class_type == box.class_type:
                iou = intersection / union
            else:
                iou = 0
            return iou

    def intersect(self, box, accuracy_mode):
        if accuracy_mode is False:
            width_inters  = self.__overlap([self.x-self.w/2, self.x+self.w/2], [box.x-box.w/2, box.x+box.w/2])
            height_inters = self.__overlap([self.y-self.h/2, self.y+self.h/2], [box.y-box.h/2, box.y+box.h/2])
            intersect_area = width_inters * height_inters
            return intersect_area

        elif accuracy_mode is True:
            width_inters  = self.__overlap([self.xmin, self.xmax], [box.xmin, box.xmax])
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

        x1 = self.x-self.w/2
        x2 = self.x+self.w/2
        x3 = box.x-box.w/2
        x4 = box.x+box.w/2

        y1 = self.y-self.h/2
        y2 = self.y+self.h/2
        y3 = box.y-box.h/2
        y4 = box.y+box.h/2

        if x2 > x4 and x1 < x3 and y2 > y4 and y1 < y3:
            return True
        else:
            return False
