import logging
from operator import itemgetter

log = logging.getLogger()


# TODO The class for the prediction should be a child of the class of the groundtruth
class BoundBox(tuple):
    # Makes the class imumutable after creation
    __slots__ = []

    def __new__(cls, xmin, xmax, ymin, ymax, probs, class_type, conf):
        area = (ymax - ymin) * (xmax - xmin)

        # Safety for avoiding weird results at network output
        if area <= 0:
            log.warn("Box has area {}".format(area))
        area = max(area, 0.0)

        return tuple.__new__(cls, (xmin, xmax, ymin, ymax, area, class_type, probs, conf))

    xmin = property(itemgetter(0))
    xmax = property(itemgetter(1))
    ymin = property(itemgetter(2))
    ymax = property(itemgetter(3))
    area = property(itemgetter(4))
    class_type = property(itemgetter(5))
    probs = property(itemgetter(6))
    conf = property(itemgetter(7))

    def iou(self, box):
        inters_xmin = max(self.xmin, box.xmin)
        inters_ymin = max(self.ymin, box.ymin)
        inters_xmax = min(self.xmax, box.xmax)
        inters_ymax = min(self.ymax, box.ymax)

        inters_width = max(0.0, inters_xmax - inters_xmin)
        inters_height = max(0.0, inters_ymax - inters_ymin)

        inters_area = inters_width * inters_height

        if inters_area <= 0:
            iou = 0.0
        else:
            try:
                iou = float(inters_area) / float(self.area + box.area - inters_area)
            except ZeroDivisionError:
                log.error("ZeroDivisionError Error")
                log.error("inters_area {}".format(inters_area))
                log.error("self.area {}".format(self.area))
                log.error("box.area {}".format(box.area))
                log.error("inters_x0 {} inters_x1 {} inters_y0 {} inters_y1 {}".format(inters_x0, inters_x1, inters_y0, inters_y1))
                raise ZeroDivisionError

        return iou
