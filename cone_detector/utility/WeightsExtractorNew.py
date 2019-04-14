import os
import tensorflow as tf
import numpy as np
import collections
from functools import reduce


def get_overflow_rate(variable, max_value):
    mask = np.logical_or(
        variable > max_value,
        variable < -max_value
    )
    overflow_rate = np.sum(mask.astype(dtype='int32'))

    return overflow_rate


def quantize(variable, width, sel_p=None):

    if sel_p is None:

        p_start = -width
        p_end = width + 1
        overflow_threshold_input = 0.0
        shape = variable.shape
        num_entries = reduce(lambda x, y: x * y, shape)
        overflow_threshold = int(overflow_threshold_input * num_entries)

        rates = []
        max_values = []
        max_value = 2.0 ** (width - 1)

        for sel_p in range(p_start, p_end-1):
            shift = 2.0 ** (width - sel_p)
            max_value_clip = max_value / shift
            rate = get_overflow_rate(variable, max_value_clip)
            rates.append(rate)
            max_values.append(max_value_clip)

        rates.append(get_overflow_rate(variable, max_value))
        max_values.append(max_value)

        rates = np.stack(rates)
        max_values = np.stack(max_values)
        rate_thresholded = rates <= overflow_threshold

        loop_p = 0
        try:
            while bool(rate_thresholded[loop_p]) is False:
                loop_p = loop_p + 1
        except IndexError:
            print('Threshold not satisfied => clipping the variable!')
            # select the max loop_p
            loop_p = loop_p - 1

        sel_p = loop_p - width
        max_value_correspondent = max_values[loop_p]

    shift_exponent = width - sel_p
    shift = 2**(width - sel_p)
    variable = variable*shift
    variable = np.clip(variable, a_min=-max_value, a_max=max_value)
    variable = variable.astype(dtype='int32', casting='unsafe')

    return variable, sel_p, max_value_correspondent, shift_exponent


if __name__ == "__main__":
    models_dir = '/home/nico/Desktop/paper_network_quantization/networks/'
    model_name = 'proteins_mixed_extreme/'
    folder_name = ''
    checkpoint_name = 'proteins_mixed_extreme-4'
    checkpoint = models_dir + model_name + folder_name +checkpoint_name
    metagraph = checkpoint + '.meta'
    output_folder = '/home/nico/Desktop/paper_network_quantization/quantized_networks/' + model_name
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    width = 8
    mixed_mode = True

    with tf.Session() as sess:
        tf.train.import_meta_graph(metagraph, clear_devices=True)
        graph = tf.get_default_graph()
        saver = tf.train.Saver()
        saver.restore(sess, checkpoint)
        collections_keys = graph.get_all_collection_keys()

        variables = tf.get_collection("trainable_variables")
        all_variables = tf.get_collection("variables")
        update_ops = tf.get_collection("update_ops")
        print(all_variables)
        variables_dict = collections.OrderedDict()
        layer_n = 0
        old_name = 'noname_noname_noname'
        for v in all_variables:
            if 'Adam' not in str(v.name) and 'step' not in str(v.name):
                v_name = str(v.name)
                print(v_name)
                # Use this for all nets except proteins
                # if ('conv' in v_name[0:13] and v_name[0:13] != old_name[0:13]) or (v_name[0:3] == 'det'):

                if ('conv' in v_name[0:4] and v_name[0:5] != old_name[0:5]) or (v_name[0:3] == 'det'):
                    print('miao')
                    layer_n += 1
                    layer_name = 'layer' + str(layer_n)
                    variables_dict[layer_name] = collections.OrderedDict()
                values = graph.get_tensor_by_name(v_name)
                variables_dict[layer_name][v_name] = collections.OrderedDict()
                variables_dict[layer_name][v_name]['name'] = v_name
                variables_dict[layer_name][v_name]['shape'] = v.shape
                variables_dict[layer_name][v_name]['values'] = sess.run(values)
                old_name = v_name

        layer_n = 0
        sel_p_w_list = []
        sel_p_b_list = []
        max_value_per_layer_weights = []
        max_value_per_layer_bias = []
        shift_per_layer_weights = []
        shift_per_layer_bias = []
        for layer in variables_dict:
            layer_n += 1

            try:
                layer_dict = variables_dict[layer]

                try:
                    weights = layer_dict['convolution_'+str(layer_n)+'_quantized_weights:0']['values']
                    gamma = layer_dict['convolution_'+str(layer_n)+'_bn/gamma:0']['values']
                    beta = layer_dict['convolution_'+str(layer_n)+'_bn/beta:0']['values']
                    moving_mean = layer_dict['convolution_'+str(layer_n)+'_bn/moving_mean:0']['values']
                    moving_variance = layer_dict['convolution_'+str(layer_n)+'_bn/moving_variance:0']['values']
                except KeyError:
                    weights = layer_dict['conv'+str(layer_n)+'_quantized_weights:0']['values']
                    gamma = layer_dict['conv'+str(layer_n)+'_bn/gamma:0']['values']
                    beta = layer_dict['conv'+str(layer_n)+'_bn/beta:0']['values']
                    moving_mean = layer_dict['conv'+str(layer_n)+'_bn/moving_mean:0']['values']
                    moving_variance = layer_dict['conv'+str(layer_n)+'_bn/moving_variance:0']['values']

                weights = weights*gamma
                weights = weights/(np.sqrt(moving_variance))

                weights_before_quantization = weights

                if mixed_mode:
                    if layer_n == 1:
                        width = 8
                    elif layer_n == 2:
                        width = 5
                    elif layer_n == 3 or layer_n == 4 or layer_n == 5:
                        width = 4
                    elif layer_n == 6 or layer_n == 7 or layer_n == 8:
                        width = 3
                    else:
                        width = 3

                weights, sel_p_w, max_value_weight_correspondent, shift = quantize(weights, width=width, sel_p=None)
                print('layer number:', layer_n)
                print('width:', width)

                sel_p_w_list.append(sel_p_w)
                max_value_per_layer_weights.append(max_value_weight_correspondent)
                shift_per_layer_weights.append(shift)

                bias1 = beta
                bias2 = gamma*moving_mean
                bias2 = bias2/(np.sqrt(moving_variance))
                bias = bias1 - bias2
                bias_before_quatization = bias
                # print(np.max(np.abs((bias))))
                bias, sel_p_b, max_value_bias_correspondent, shift = quantize(bias, width=width, sel_p=None)

                sel_p_b_list.append(sel_p_b)
                max_value_per_layer_bias.append(max_value_bias_correspondent)
                shift_per_layer_bias.append(shift)

                np.set_printoptions(threshold=np.nan)

                # np.save(output_folder+'not_quatized_weights_layer'+str(layer_n), weights_before_quantization)
                # np.save(output_folder+'not_quatized_weights_layer'+str(layer_n), bias_before_quatization)
                # print(weights)
                # exit()
                np.save(output_folder+'layer'+str(layer_n)+'_weights', weights)
                np.save(output_folder+'layer'+str(layer_n)+'_biases', bias)

            except KeyError:

                if mixed_mode:
                    width = 16

                weights = layer_dict['det_q_quantized_weights:0']['values']
                weights, sel_p_w, max_value_weight_correspondent, shift = quantize(weights, width=width, sel_p=None)
                print('detector_layer')
                print('width:', width)
                sel_p_w_list.append(sel_p_w)
                max_value_per_layer_weights.append(max_value_weight_correspondent)
                shift_per_layer_weights.append(shift)

                bias = layer_dict['Variable:0']['values']
                bias, sel_p_b, max_value_bias_correspondent, shift = quantize(bias, width=width, sel_p=None)

                sel_p_b_list.append(sel_p_b)
                max_value_per_layer_bias.append(max_value_bias_correspondent)
                shift_per_layer_bias.append(shift)

                np.save(output_folder+'layer'+str(9)+'_weights', weights)
                np.save(output_folder+'layer'+str(9)+'_biases', bias)

        print('The weights selected p\n', sel_p_w_list)
        print('The correspondent shift\n', [shift for shift in shift_per_layer_weights])
        print('The correspondent point position\n', [sel_p - width for sel_p in sel_p_w_list])
        print('The max representable value per weights layer \n', max_value_per_layer_weights)
        print('\n')
        print('The biases selected p\n', sel_p_b_list)
        print('The correspondent shift\n', [shift for shift in shift_per_layer_bias])
        print('The correspondent point position\n', [sel_p - width for sel_p in sel_p_b_list])
        print('The max representable value per bias layer \n', max_value_per_layer_bias)

        np.save(output_folder+'/max_value_per_layer_weights', max_value_per_layer_weights)
        np.save(output_folder+'max_value_per_layer_bias', max_value_per_layer_bias)

        np.save(output_folder+'/shift_per_layer_weights', shift_per_layer_weights)
        np.save(output_folder+'shift_per_layer_bias', shift_per_layer_bias)

        np.save(output_folder+'sel_p_w', sel_p_w_list)
        np.save(output_folder+'sel_p_b', sel_p_b_list)





        for op in graph.get_operations():
            name = str(op.name)
            #
            # if (op.type == "Conv2D"):
            #     print(op)
            #     print('...')

            # print(op.name)
            if 'MaxPool' in name or 'relu' in name or 'quantized_weights' in name:
                if 'MaxPoolGrad' not in name and 'ReluGrad' not in name and 'clip' not in name and 'Adam' not in name and 'Assign' not in name and 'read' not in name and 'round' not in name:
                    # print(op.name)
                    pass