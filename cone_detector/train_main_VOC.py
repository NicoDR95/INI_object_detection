import logging
import tensorflow as tf
from Parameters import Parameters
from training.TrainPipeline import TrainPipeline
from accuracy.Accuracy import Accuracy
from datahandling.BatchGenerator import BatchGenerator
from datahandling.DataPreprocessing import DataPreprocessing
from datahandling.Dataset import Dataset

from predict.Predict import Predict
from training.Optimizer import Optimizer
from training.YoloLossCrossEntropyProb import YoloLossCrossEntropyProb
from visualization.Visualization import Visualization
from datahandling.DataAugmentation import DataAugmentation
from utility.CalculateAnchors import CalculateAnchors
from utility.DatasetTFRecordsConverter import TFRecordsConverter

from networks.TinyYoloOnProteins import TinyYoloOnProteins
from networks.MemlessNet import MemlessNet


logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()

# ~~~~~~~~~ Directories for training ~~~~~~~~~
run_name = 'VOC_test'
run_index = 2

ws_root = "/home/nico/semester_project/cone_detector_data/"
data_root = "/home/nico/semester_project/VOCdevkit/VOC2012/"
images_dir = data_root + 'test_img/'
annotations_dir = data_root + 'test_ann/'
saved_model_dir = ws_root + 'saved_models/' + run_name + '/run_{}/'.format(run_index)
aug_annotations_dir = data_root + 'empty/'          # point to empty or inexistent folder if you don't want to use augmented data
aug_images_dir = data_root + 'augmented_images/'
all_images_dir = [images_dir, aug_images_dir]

# ~~~~~~~~~ Directories for inference ~~~~~~~~~
checkpoint_number = '4'
saved_model_name = run_name
checkpoint = saved_model_dir + saved_model_name + '-' + checkpoint_number
metagraph = checkpoint + '.meta'

# ~~~~~~~~~ Directories for validation ~~~~~~~~~
# validation_data_dir = ws_root + 'validation/'
validation_data_dir = data_root
validation_images_dir = validation_data_dir + 'val_img/'
validation_annotations_dir = validation_data_dir + 'val_ann/'

# ~~~~~~~~~ Directories for augmentation ~~~~~~~~~
augmented_image_dir = data_root + 'augmented_images/'
augmented_annotations_dir = data_root + 'augmented_annotations/'
tfrecord_output_dir = data_root + 'tfrecords_output/'

# ~~~~~~~~~ Directories for testing ~~~~~~~~~
test_dir = ws_root + 'test/'
test_image_dir = test_dir + 'test_image/'
test_image_name = '400543.jpg'
test_image_path = test_image_dir + test_image_name
test_video_dir = test_dir + 'test_video/'
test_video_name = 'trackdrive_cropped.mp4'
test_video_path = test_video_dir + test_video_name

# ~~~~~~~~~ Directories for anchors ~~~~~~~~~
annotations_for_anchors = annotations_dir

# ~~~~~~~~~ General settings ~~~~~~~~~
training_mode = True
inference_mode = False
validation_mode = False
augmentation_mode = False
anchors_mode = False

# ~~~~~~~~~ Training settings ~~~~~~~~~
save_as_graphdef = False
visualize_dataset = False
visualize_preprocessed_images = False
leaky_relu = False
use_sqrt_loss = False
checkpoints_to_keep = 10        # number of chkp you want to keep at any time, older are automatically deleted
# labels_list = ['yellow_cones', 'blue_cones', 'orange_cones']
labels_list = ['person', 'car', 'bicycle', 'bus', 'motorbike', 'train', 'aeroplane', 'boat', 'chair', 'bottle',
               'diningtable', 'pottedplant', 'tvmonitor', 'sofa', 'bird', 'cat', 'cow', 'dog', 'horse', 'sheep', 'head', 'hand', 'foot']
# anchors = [0.86, 1.69, 1.44, 2.96, 0.35, 0.66, 2.34, 4.91]
anchors = [1.3221, 1.73145, 3.19275, 4.00944, 5.05587, 8.09892, 9.47112, 4.84053, 11.2364, 10.0071]
loss_filename_print_threshold = 50
fixed_point_width = 8
n_classes = 23
n_anchors = int(len(anchors)/2)
input_h = 416
input_w = 416
add_fourth_channel = False
use_grayscale_mask = False
use_hue_mask = True
visualize_fourth_channel = False            # Use this to visualize if the result on the 4th channel are good
input_depth = 3
output_h = 13
output_w = 13
batch_size = 1
n_epochs = 10000
scale_coor = 1.0
scale_noob = 1.0
scale_conf = 5.0
scale_proob = 1.0
data_preprocessing_normalize = 255.0
tf_device = "/gpu:0"
debug = True
print_sel_p=False
learning_rate = 10 ** (-6)
dropout = [0.0]

# ~~~~~~~~~ Inference and accuracy settings ~~~~~~~~~
import_graph_from_metafile = True       # set to true if you intend to run inference on a graph in a meta file
weights_from_npy = False                     # set to true only if using a graph that loads weights from npy files
keep_small_ones = False                      # to avoid avaing big boxes with more cones in one
car_pov_inference_mode = False
min_distance_from_top = 200                 # put 0 to disenable the contidion
max_area_distant_obj = 18000               # Put an extremely high value to disenable the condition
video_mode = True
save_video_to_file = False
framerate = 20
video_output_name = 'trackdrive_quantized_p8_cross' + '_' + str(framerate)
fsg_accuracy_mode = False
visualize_accuracy_outputs = False
threshold = 0.7
iou_threshold = 0.4   # Threshold used for non-max suppression (The higher it is, the lower the suppression)
iou_accuracy_thr = 0.4  # Threshold used for accuracy metric (if iou with ground truth is higher then it's a TP)

# ~~~~~~~~~ Data augmentation settings ~~~~~~~~~
augmentation_run = '11'                  # Change this number to perform augmentation runs without replacing old ones
visualize_augmented_dataset_mode = False
make_tfrecords_mode = True
check_augmentation = False
crop_for_aspect_ratio = True
target_aspect_ratio = 2                 # 2:1
shuffle_functions = True
mirror_flag = False
translate_flag = False
translate_x = 100
translate_y = 100
rotate_flag = False
rotate_range = 6                      # The rotation angle on each image will be a random angle
rotate_no_cut = True                    # between +- rotate_range
illumination_flag = False
gamma_change_min = 0.7
gamma_change_max = 1.7
gaussian_noise_flag = True
gaussian_mean = 0
gaussian_standard_dev_min = 11
gaussian_standard_dev_max = 17
salt_pepper_noise_flag = False
salt_vs_pepper_min = 0.3
salt_vs_pepper_max = 0.7
salt_pepper_amount_min = 0.003
salt_pepper_amount_max = 0.008
poisson_noise_flag = False               # quite useless, r wrong implementation
speckle_noise_flag = False               # this adds an extreme amount of noise, don't use it
gaussian_blur_flag = False
gaussian_kernel_min = 7                 # must be odd numbers
gaussian_kernel_max = 15
gaussian_sigma = 0                       # leave it to 0 so that it's calculated from the the kernel size
average_blur_flag = False                # don't use both blur together
average_kernel_min = 11                  # must be odd numbers
average_kernel_max = 21

# ~~~~~~~~~ Anchors mode settings ~~~~~~~~~
n_clusters = 4                  # anchors mode is run on annotations_dir files
n_init = 1000                   # Number of time the k-means algorithm will be run with different centroid seeds
max_iter = 1000                 # Maximum number of iterations of the k-means algorithm for a single run.

# ~~~~~~~~~ Class settings ~~~~~~~~~
parameters_type = Parameters
dataset_parser_type = Dataset
network_type = TinyYoloOnProteins
data_preprocessor_type = DataPreprocessing
batch_generator_type = BatchGenerator
loss_type = YoloLossCrossEntropyProb
optimizer_type = Optimizer
pipeline_type = TrainPipeline
visualizer_type = Visualization
predictor_type = Predict
accuracy_type = Accuracy
augmentation_type = DataAugmentation
anchors_calculator_type = CalculateAnchors
tfrecord_converter_type = TFRecordsConverter

if __name__ == "__main__":
    with tf.device(tf_device):
        train_parameters = parameters_type(n_classes=n_classes,
                                           n_anchors=n_anchors,
                                           anchors=anchors,
                                           input_h=input_h,
                                           input_w=input_w,
                                           input_depth=input_depth,
                                           output_h=output_h,
                                           output_w=output_w,
                                           annotations_dir=annotations_dir,
                                           images_dir=images_dir,
                                           all_images_dir=all_images_dir,
                                           saved_model_dir=saved_model_dir,
                                           saved_model_name=saved_model_name,
                                           training=training_mode,
                                           augmentation_mode=augmentation_mode,
                                           batch_size=batch_size,
                                           n_epochs=n_epochs,
                                           scale_coor=scale_coor,
                                           scale_noob=scale_noob,
                                           scale_conf=scale_conf,
                                           scale_proob=scale_proob,
                                           dropout=dropout,
                                           learning_rate=learning_rate,
                                           data_preprocessing_normalize=data_preprocessing_normalize,
                                           labels_list=labels_list,
                                           debug=debug,
                                           threshold=threshold,
                                           iou_threshold=iou_threshold,
                                           metagraph=metagraph,
                                           checkpoint=checkpoint,
                                           visualize_dataset=visualize_dataset,
                                           visualize_preprocessed_images=visualize_preprocessed_images,
                                           video_mode=video_mode,
                                           test_image_path=test_image_path,
                                           test_video_path=test_video_path,
                                           leaky_relu=leaky_relu,
                                           use_sqrt_loss=use_sqrt_loss,
                                           validation_images_dir=validation_images_dir,
                                           validation_annotations_dir=validation_annotations_dir,
                                           iou_accuracy_thr=iou_accuracy_thr,
                                           augmented_image_dir=augmented_image_dir,
                                           augmented_annotations_dir=augmented_annotations_dir,
                                           augmentation_run=augmentation_run,
                                           fsg_accuracy_mode=fsg_accuracy_mode,
                                           visualize_accuracy_outputs=visualize_accuracy_outputs,
                                           add_fourth_channel=add_fourth_channel,
                                           use_grayscale_mask=use_grayscale_mask,
                                           use_hue_mask=use_hue_mask,
                                           visualize_fourth_channel=visualize_fourth_channel,
                                           import_graph_from_metafile=import_graph_from_metafile,
                                           checkpoints_to_keep=checkpoints_to_keep,
                                           save_video_to_file=save_video_to_file,
                                           framerate=framerate,
                                           loss_filename_print_threshold=loss_filename_print_threshold,
                                           tf_device=tf_device,
                                           save_as_graphdef=save_as_graphdef,
                                           video_output_name=video_output_name,
                                           min_distance_from_top=min_distance_from_top,
                                           max_area_distant_obj=max_area_distant_obj,
                                           car_pov_inference_mode=car_pov_inference_mode,
                                           keep_small_ones=keep_small_ones,
                                           weights_from_npy=weights_from_npy,
                                           fixed_point_width=fixed_point_width,
                                           print_sel_p=print_sel_p
                                           )

        dataset_parser = dataset_parser_type(parameters=train_parameters,
                                             annotations_dir=annotations_dir)
        aug_dataset_parser = dataset_parser_type(parameters=train_parameters,
                                                 annotations_dir=aug_annotations_dir)
        cones_dataset_parser = [dataset_parser, aug_dataset_parser]     # Put here ll the dataset you intend to train on toghether
        single_dataset_parser = dataset_parser      # Put here the single set on which you want to perform aumentation or acccuracy or anchors k mean

        data_preprocessor = data_preprocessor_type(parameters=train_parameters)

        batch_generator = batch_generator_type(parameters=train_parameters)

        yolo_network = network_type(parameters=train_parameters)

        yolo_loss = loss_type(parameters=train_parameters)

        train_optimizer = optimizer_type(parameters=train_parameters)

        predictor = predictor_type(parameters=train_parameters,
                                   preprocessor=data_preprocessor,
                                   network=yolo_network)

        visualize = visualizer_type(parameters=train_parameters,
                                    preprocessor=data_preprocessor,
                                    prediction=predictor)

        accuracy = accuracy_type(parameters=train_parameters,
                                 prediction=predictor,
                                 dataset=single_dataset_parser,
                                 preprocessor=data_preprocessor,
                                 visualize=visualize)

        anchors_calculators = anchors_calculator_type(parameters=train_parameters,
                                                      dataset=single_dataset_parser,
                                                      annotations_dir=annotations_for_anchors,
                                                      n_clusters=n_clusters,
                                                      n_init=n_init,
                                                      max_iter=max_iter)
        tfrecord_converter = tfrecord_converter_type(parameters=train_parameters,
                                                     dataset=single_dataset_parser,
                                                     data_preprocessing=data_preprocessor,
                                                     output_path=tfrecord_output_dir)

        augmentation = augmentation_type(parameters=train_parameters,
                                         image_dir=images_dir,
                                         annotations_dir=annotations_dir,
                                         augmented_image_dir=augmented_image_dir,
                                         augmented_annotations_dir=augmented_annotations_dir,
                                         dataset=single_dataset_parser,
                                         visualization=visualize,
                                         check_augmentation=check_augmentation,
                                         crop_for_aspect_ratio=crop_for_aspect_ratio,
                                         target_aspect_ratio=target_aspect_ratio,
                                         shuffle_functions=shuffle_functions,
                                         mirror=mirror_flag,
                                         translate=translate_flag,
                                         translate_x=translate_x,
                                         translate_y=translate_y,
                                         rotate=rotate_flag,
                                         rotate_range=rotate_range,
                                         rotate_no_cut=rotate_no_cut,
                                         illumination=illumination_flag,
                                         gamma_change_min=gamma_change_min,
                                         gamma_change_max=gamma_change_max,
                                         gaussian_noise=gaussian_noise_flag,
                                         gaussian_mean=gaussian_mean,
                                         gaussian_standard_dev_min=gaussian_standard_dev_min,
                                         gaussian_standard_dev_max=gaussian_standard_dev_max,
                                         salt_pepper_noise=salt_pepper_noise_flag,
                                         salt_vs_pepper_min=salt_vs_pepper_min,
                                         salt_vs_pepper_max=salt_vs_pepper_max,
                                         salt_pepper_amount_min=salt_pepper_amount_min,
                                         salt_pepper_amount_max=salt_pepper_amount_max,
                                         poisson_noise=poisson_noise_flag,
                                         speckle_noise=speckle_noise_flag,
                                         gaussian_blur=gaussian_blur_flag,
                                         gaussian_kernel_min=gaussian_kernel_min,
                                         gaussian_kernel_max=gaussian_kernel_max,
                                         gaussian_sigma=gaussian_sigma,
                                         average_blur=average_blur_flag,
                                         average_kernel_min=average_kernel_min,
                                         average_kernel_max=average_kernel_max

                                         )

        train_pipeline = pipeline_type(parameters=train_parameters,
                                       dataset=cones_dataset_parser,
                                       data_preprocessing=data_preprocessor,
                                       batch_generator=batch_generator,
                                       network=yolo_network,
                                       loss=yolo_loss,
                                       optimizer_object=train_optimizer,
                                       visualizer=visualize,
                                       accuracy=accuracy)

        if inference_mode is False and training_mode is True and validation_mode is False and augmentation_mode is False and anchors_mode is False:
            train_pipeline.train()
        elif inference_mode is True and training_mode is False and validation_mode is False and augmentation_mode is False and anchors_mode is False:
            visualize.run_net_and_get_predictions()
        elif inference_mode is False and training_mode is False and validation_mode is True and augmentation_mode is False and anchors_mode is False:
            accuracy.run_and_get_accuracy()
        elif inference_mode is False and training_mode is False and validation_mode is False and augmentation_mode is True and anchors_mode is False:
            if visualize_augmented_dataset_mode is True:
                augmentation.visualize_augmented_dataset()
            elif make_tfrecords_mode is True:
                tfrecord_converter.convert_dataset()
            else:
                augmentation.data_aug_pipeline()
        elif inference_mode is False and training_mode is False and validation_mode is False and augmentation_mode is False and anchors_mode is True:
            anchors_calculators.get_anchors()
        else:
            raise ValueError("Invalid combination of train, inference, augmentation and validations modes!")