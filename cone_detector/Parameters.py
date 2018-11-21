class Parameters(object):
    def __init__(self,
                 n_classes,
                 n_anchors,
                 anchors,
                 input_h,
                 input_w,
                 input_depth,
                 output_h,
                 output_w,
                 annotations_dir,
                 images_dir,
                 all_images_dir,
                 saved_model_dir,
                 saved_model_name,
                 training,
                 augmentation_mode,
                 batch_size,
                 n_epochs,
                 learning_rate,
                 dropout,
                 scale_coor,
                 scale_noob,
                 scale_conf,
                 scale_proob,
                 data_preprocessing_normalize,
                 labels_list,
                 debug,
                 conf_threshold,
                 iou_threshold,
                 metagraph,
                 checkpoint,
                 visualize_dataset,
                 visualize_preprocessed_images,
                 video_mode,
                 test_image_path,
                 test_video_path,
                 leaky_relu,
                 use_sqrt_loss,
                 validation_images_dir,
                 validation_annotations_dir,
                 iou_accuracy_thr,
                 augmented_image_dir,
                 augmented_annotations_dir,
                 augmentation_run,
                 fsg_accuracy_mode,
                 visualize_accuracy_outputs,
                 add_fourth_channel,
                 use_grayscale_mask,
                 use_hue_mask,
                 visualize_fourth_channel,
                 import_graph_from_metafile,
                 checkpoints_to_keep,
                 save_video_to_file,
                 framerate,
                 loss_filename_print_threshold,
                 tf_device,
                 save_as_graphdef,
                 video_output_name,
                 min_distance_from_top,
                 max_area_distant_obj,
                 car_pov_inference_mode,
                 keep_small_ones,
                 weights_from_npy,
                 fixed_point_width,
                 print_sel_p
                 ):
        self.n_classes = n_classes
        self.n_anchors = n_anchors
        self.anchors = anchors
        self.input_h = input_h
        self.input_w = input_w
        self.input_depth = input_depth
        self.output_h = output_h
        self.output_w = output_w
        self.annotations_dir = annotations_dir
        self.images_dir = images_dir
        self.all_images_dir = all_images_dir
        self.saved_model_dir = saved_model_dir
        self.saved_model_name = saved_model_name
        self.tensorboard_dir = self.saved_model_dir + 'tensorboard'
        self.training = training
        self.augmentation_mode = augmentation_mode
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.scale_coor = scale_coor
        self.scale_noob = scale_noob
        self.scale_conf = scale_conf
        self.scale_proob = scale_proob
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.data_preprocessing_normalize = data_preprocessing_normalize
        self.labels_list = labels_list
        self.debug = debug
        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.metagraph = metagraph
        self.checkpoint = checkpoint
        self.visualize_dataset = visualize_dataset
        self.visualize_preprocessed_images = visualize_preprocessed_images
        self.video_mode = video_mode
        self.test_image_path = test_image_path
        self.test_video_path = test_video_path
        self.leaky_relu = leaky_relu
        self.use_sqrt_loss = use_sqrt_loss
        self.validation_images_dir = validation_images_dir
        self.validation_annotations_dir = validation_annotations_dir
        self.iou_accuracy_thr = iou_accuracy_thr
        self.augmented_image_dir = augmented_image_dir
        self.augmented_annotations_dir = augmented_annotations_dir
        self.augmentation_run = augmentation_run
        self.fsg_accuracy_mode = fsg_accuracy_mode
        self.visualize_accuracy_outputs = visualize_accuracy_outputs
        self.add_fourth_channel = add_fourth_channel
        self.use_grayscale_mask = use_grayscale_mask
        self.use_hue_mask = use_hue_mask
        self.visualize_fourth_channel = visualize_fourth_channel
        self.import_graph_from_metafile = import_graph_from_metafile
        self.checkpoints_to_keep = checkpoints_to_keep
        self.save_video_to_file = save_video_to_file
        self.framerate = framerate
        self.loss_filename_print_threshold = loss_filename_print_threshold
        self.save_as_graphdef = save_as_graphdef
        self.video_output_name = video_output_name
        self.min_distance_from_top = min_distance_from_top
        self.max_area_distant_obj = max_area_distant_obj
        self.car_pov_inference_mode = car_pov_inference_mode
        self.keep_small_ones = keep_small_ones
        self.weights_from_npy = weights_from_npy
        self.fixed_point_width = fixed_point_width
        self.print_sel_p = print_sel_p
        self.tf_device = tf_device
        self.true_values_shape = [None, output_h, output_w, n_anchors, 4 + 1 + n_classes + 2]
