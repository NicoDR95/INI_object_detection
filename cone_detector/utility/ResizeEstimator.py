import cv2
import os

# Only setup to do is pointing to the folder containing the images
images_path = "/home/asa/workspaces/Pycharm/yolo/dataset/VOC2012/JPEGImages/"

#
#
##
# Code start
##

all_sizes = list()
files = os.listdir(images_path)

print("Starting read of {} images".format(len(files)))
for filename in files:
    full_image_path = images_path + filename

    cv2_image = cv2.imread(full_image_path)

    size = cv2_image.shape

    all_sizes.append(size)

total_width = 0
total_height = 0
total_aspect_ratio = 0
for (height, width, depth) in all_sizes:
    total_width = total_width + width
    total_height = total_height + height
    total_aspect_ratio = total_aspect_ratio + width / height

num_images = len(all_sizes)
average_width = total_width / num_images
average_height = total_height / num_images
average_aspect_ratio = total_aspect_ratio / num_images

print("Average width: {:.2f} Average height: {:.2f} Average aspect ratio: {:.2f} ({} images)".format(average_width, average_height,
                                                                                                  average_aspect_ratio,
                                                                                                  num_images))
