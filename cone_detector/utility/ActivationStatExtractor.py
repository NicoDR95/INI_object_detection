import os
import cv2
import tensorflow as tf
import numpy as np
from utility.WeightsExtractorNew import quantize


def preprocess(test_image_path):
    image_test = cv2.imread(test_image_path)
    resized_image = cv2.resize(image_test, (input_w, input_h))
    resized_image = np.array(resized_image, dtype=np.float32)
    normalized_image = resized_image / 256
    return normalized_image


if __name__ == "__main__":
    test_images_folder = '/home/nico/Desktop/paper_network_quantization/test_images/'
    models_dir = '/home/nico/Desktop/paper_network_quantization/networks/'
    model_name = 'proteins_mixed_extreme/'
    folder_name = ''
    checkpoint_name = 'proteins_mixed_extreme-4'
    checkpoint = models_dir + model_name + folder_name + checkpoint_name
    metagraph = checkpoint + '.meta'
    output_folder = '/home/nico/Desktop/paper_network_quantization/quantized_networks/' + model_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    width = 16
    input_w = 512
    input_h = 256

    with tf.Session() as sess:
        # Load and restore checkpoint
        tf.train.import_meta_graph(metagraph, clear_devices=True)
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        graph = tf.get_default_graph()

        # Main placeholders
        image_ph = graph.get_tensor_by_name("image_placeholder:0")
        train_flag_ph = graph.get_tensor_by_name("flag_placeholder:0")
        output_node_ph = graph.get_tensor_by_name('network_output:0')

        # Intermediate layer activation_ph
        try:
            layer_2_act = graph.get_tensor_by_name("convolution_2:0")
            layer_3_act = graph.get_tensor_by_name("convolution_3:0")
            layer_4_act = graph.get_tensor_by_name("convolution_4:0")
            layer_5_act = graph.get_tensor_by_name("convolution_5:0")
            layer_6_act = graph.get_tensor_by_name("convolution_6:0")
            layer_7_act = graph.get_tensor_by_name("convolution_7:0")
            layer_8_act = graph.get_tensor_by_name("convolution_8:0")
            layer_9_act = graph.get_tensor_by_name("det_q:0")
            # MaxPool layer inputs
            layer_1_maxpool_input = graph.get_tensor_by_name("relu:0")
            layer_2_maxpool_input = graph.get_tensor_by_name("relu_1:0")
            layer_3_maxpool_input = graph.get_tensor_by_name("relu_2:0")
            layer_4_maxpool_input = graph.get_tensor_by_name("relu_3:0")
            layer_5_maxpool_input = graph.get_tensor_by_name("relu_4:0")
        except:

            layer_2_act = graph.get_tensor_by_name("conv2:0")
            layer_3_act = graph.get_tensor_by_name("conv3:0")
            layer_4_act = graph.get_tensor_by_name("conv4:0")
            layer_5_act = graph.get_tensor_by_name("conv5:0")
            layer_6_act = graph.get_tensor_by_name("conv6:0")
            layer_7_act = graph.get_tensor_by_name("conv7:0")
            layer_8_act = graph.get_tensor_by_name("conv8:0")
            layer_9_act = graph.get_tensor_by_name("det_q:0")
            # MaxPool layer inputs
            layer_1_maxpool_input = graph.get_tensor_by_name("relu:0")
            layer_2_maxpool_input = graph.get_tensor_by_name("relu_1:0")
            layer_3_maxpool_input = graph.get_tensor_by_name("relu_2:0")
            layer_4_maxpool_input = graph.get_tensor_by_name("relu_3:0")
            layer_5_maxpool_input = graph.get_tensor_by_name("relu_4:0")



        tensors_to_fetch = [layer_2_act,
                            layer_3_act,
                            layer_4_act,
                            layer_5_act,
                            layer_6_act,
                            layer_7_act,
                            layer_8_act,
                            layer_9_act,
                            output_node_ph]

        more_tensors_to_fetch = [layer_1_maxpool_input,
                                 layer_2_maxpool_input,
                                 layer_3_maxpool_input,
                                 layer_4_maxpool_input,
                                 layer_5_maxpool_input]

        batch_input = []
        for test_image in sorted(os.listdir(test_images_folder)):
            test_image_path = test_images_folder + test_image
            test_image_preprocessed = preprocess(test_image_path)
            batch_input.append(test_image_preprocessed)
        batch_input = np.stack(batch_input)

        network_output = sess.run(fetches=tensors_to_fetch,
                                  feed_dict={image_ph: batch_input,
                                             train_flag_ph: False})
        max_pool_inputs = sess.run(fetches=more_tensors_to_fetch,
                                   feed_dict={image_ph: batch_input,
                                              train_flag_ph: False})

        activation_shift_list = []
        activation, sel_p_a, max_value_activation_correspondent, shift = quantize(batch_input, width=16, sel_p=None)
        activation_shift_list.append(shift)

        for l_count, layer_act in enumerate(network_output[0:-1]):
            # np.save(output_folder + 'layer_' + str(l_count + 2) + '_activation', layer_act)

            activation, sel_p_a, max_value_activation_correspondent, shift = quantize(layer_act, width=width,
                                                                                      sel_p=None)

            zero_amount = np.count_nonzero(activation == 0)
            all_elements_amount = activation.size
            sparsity = zero_amount / all_elements_amount
            print('layer {} activation sparsity is: {}'.format(l_count + 2, sparsity))
            activation_shift_list.append(shift)

        np.save(output_folder + 'shift_per_layer_activations', activation_shift_list)
        print('\n')
        print('The activation shifts are:')
        print(activation_shift_list)
        print('\n')

        for l_count, max_pool_input in enumerate(max_pool_inputs):
            # np.save(output_folder + 'layer_' + str(l_count + 1) + '_max_pool_input', max_pool_input)
            max_pool_input, sel_p_a, max_value_activation_correspondent, shift = quantize(max_pool_input, width=width,
                                                                                          sel_p=None)

            zero_amount = np.count_nonzero(max_pool_input == 0)
            all_elements_amount = max_pool_input.size
            sparsity = zero_amount / all_elements_amount
            print('layer {} max_pool sparsity is: {}'.format(l_count + 1, sparsity))
