import tensorflow as tf
import sys
if sys.version_info.major >= 3:
    import pathlib
else:
    import pathlib2 as pathlib

frozen_dir = '/home/nico/semester_project/cone_detector_data/saved_models/tiny-yolov2/saved_model_11_TinyYoloOnProteins/frozen/'

# converter = tf.contrib.lite.TocoConverter.from_saved_model(frozen_dir)
# converter.post_training_quantize = True
# quantized_model = converter.convert()
# open(saved_model_dir + 'quantized_model.tflite', "wb").write(quantized_model)

graph_def_file = pathlib.Path(frozen_dir + "frozen_graph.pb")
# graph_def_file = frozen_dir + 'frozen_graph.pb'
input_arrays = ["image_placeholder"]
output_arrays = ["network_output"]
converter = tf.contrib.lite.TocoConverter.from_frozen_graph(
  str(graph_def_file), input_arrays, output_arrays, input_shapes={"input": [1, 256, 512, 3]})
converter.post_training_quantize = True
resnet_tflite_file = graph_def_file.parent/"TinyYolo_quantized.tflite"
# resnet_tflite_file = frozen_dir + "TinyYolo_quantized.tflite"
resnet_tflite_file.write_bytes(converter.convert())