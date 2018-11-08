import cv2
import os
import re

video_dir = '/home/nico/semester_project/cone_detector_data/test/test_video/'
video_name = 'trackdrive_cropped.mp4'
video_path = video_dir + video_name
video_name_output = 'FsgYoloOutput.avi'
video_path_output = video_dir + video_name_output
fsgyolo_dir = '/home/nico/semester_project/FSG_YOLO/pytorch-yolo2-marvis/'
temp_work_dir = '/home/nico/semester_project/FSG_YOLO/pytorch-yolo2-marvis/temp/'
images_out_dir = '/home/nico/semester_project/FSG_YOLO/pytorch-yolo2-marvis/output_images/'
framerate = 6

# Transform video in images
# cap = cv2.VideoCapture(video_path)
# frame = 0
# while cap.isOpened():
#     ret, image = cap.read()
#     frame += 1
#     if ret is not True:
#         break
#     cv2.imwrite(temp_work_dir + str(frame) + '.jpg', image)
#
# # Get output images from network
# os.chdir(fsgyolo_dir)
# for image in os.listdir(temp_work_dir):
#     os.system('python2 detect.py cfg/yolo-voc-5-class-and-anchors.cfg 000400_25June2018_smallBoxes.weights ' + temp_work_dir + image)

# Make the images into a video
fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_path_output, fourcc, framerate, (1600, 800))
list_of_images = []
for image_name in os.listdir(images_out_dir):
    list_of_images.append(image_name)
convert = lambda text: float(text) if text.isdigit() else text
alphanum = lambda key: [ convert(c) for c in re.split('([-+]?[0-9]*\.?[0-9]*)', key)]
list_of_images.sort(key=alphanum)
for image_name in list_of_images:
    print(image_name)
    image = cv2.imread(images_out_dir + image_name)
    video.write(image)
video.release()

