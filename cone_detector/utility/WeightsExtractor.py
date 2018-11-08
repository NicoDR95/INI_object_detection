import tensorflow as tf
import numpy as np
import collections
from functools import reduce


models_dir = '/home/nico/semester_project/cone_detector_data/saved_models/'
model_name = 'TinyYoloOnProteins_cross_entropy/'
folder_name = 'saved_model_15_quantized_p8/'
checkpoint_name = 'TinyYoloOnProteins_cross_entropy-25'
checkpoint = models_dir + model_name + folder_name +checkpoint_name
metagraph = checkpoint + '.meta'

width = 8


def get_overflow_rate(variable, max_value):
    mask = np.logical_or(
        variable > max_value,
        variable < -max_value
    )

    overflow_rate = np.sum(mask.astype(dtype='int32'))

    return overflow_rate


def quantize(variable, sel_p=None):
    # print('###########')
    # print(variable)
    if sel_p is None:

        p_start = -width
        p_end = width + 1
        overflow_threshold_input = 0.0
        shape = variable.shape
        num_entries = reduce(lambda x, y: x * y, shape)
        overflow_threshold = int(overflow_threshold_input * num_entries)

        rates = []
        shifts = []

        max_value = 2.0 ** (width - 1)

        for point in range(p_start, p_end):
            shift = 2.0 ** (width - point)

            max_value_clip = max_value / shift

            rate = get_overflow_rate(variable, max_value_clip)

            rates.append(rate)
            shifts.append(shift)

        while_loop_iter = len(rates)
        rates = np.stack(rates)
        shifts = np.stack(shifts)

        rate_thresholded = rates <= overflow_threshold      # returns a boolean tensor

        loop_p = 0

        while bool(rate_thresholded[loop_p]) is False:
            loop_p = loop_p + 1

        sel_p = loop_p
        # print(sel_p)

    point = -width + sel_p
    shift = 2**(width - point)
    # max_val = 2**15/shift
    # min_val = -max_val
    variable = variable*shift
    # print('###########')
    # print(variable)
    # array = np.around(array, decimals=0)
    # array = np.clip(array, a_min=min_val, a_max=max_val)
    variable = variable.astype(dtype='int16', casting='unsafe')
    # print('###########')
    # print(variable)

    return variable, sel_p


with tf.Session() as sess:
    tf.train.import_meta_graph(metagraph, clear_devices=True)
    graph = tf.get_default_graph()
    saver = tf.train.Saver()
    saver.restore(sess, checkpoint)
    collections_keys = graph.get_all_collection_keys()

    variables = tf.get_collection("trainable_variables")
    all_variables = tf.get_collection("variables")
    update_ops = tf.get_collection("update_ops")

    # for v in all_variables:
    #     if 'Adam' not in str(v.name):
    #         print(v)
    # for op in update_ops:
    #     op_value = graph.get_tensor_by_name(op)
    #     print(sess.run(op_value))

    variables_dict = collections.OrderedDict()
    layer_n = 0
    old_name = 'noname'
    for v in all_variables:
        if 'Adam' not in str(v.name):
            v_name = str(v.name)
            if (v_name[0:4] == 'conv' and v_name[0:5] != old_name[0:5]) or (v_name[0:3] == 'det'):
                layer_n += 1
                layer_name = 'layer' + str(layer_n)
                variables_dict[layer_name] = collections.OrderedDict()
            # print(v)
            values = graph.get_tensor_by_name(v_name)
            variables_dict[layer_name][v_name] = collections.OrderedDict()
            variables_dict[layer_name][v_name]['name'] = v_name
            variables_dict[layer_name][v_name]['shape'] = v.shape
            variables_dict[layer_name][v_name]['values'] = sess.run(values)
            old_name = v_name
    # print(variables_dict)

    layer_n = 0
    sel_p_w_list = []
    sel_p_b_list = []
    for layer in variables_dict:
        layer_n += 1
        if layer_n == 5 or layer_n == 6 or layer_n == 7 or layer_n == 8:
            p_weights = 5
        elif layer_n == 1 or layer_n == 2 or layer_n == 9:
            p_weights = 7
        elif layer_n == 3 or layer_n == 4:
            p_weights = 6
        p_bias = 8
        # print(layer)
        try:
            layer_dict = variables_dict[layer]
            weights = layer_dict['conv'+str(layer_n)+'_quantized_weights:0']['values']
            gamma = layer_dict['conv'+str(layer_n)+'_bn/gamma:0']['values']
            beta = layer_dict['conv'+str(layer_n)+'_bn/beta:0']['values']
            moving_mean = layer_dict['conv'+str(layer_n)+'_bn/moving_mean:0']['values']
            moving_variance = layer_dict['conv'+str(layer_n)+'_bn/moving_variance:0']['values']

            weights = weights*gamma
            weights = weights/(np.sqrt(moving_variance))
            # print(weights)
            # print(p_weights)
            # print(np.amax(weights))

            weights, sel_p_w = quantize(weights, sel_p=None)
            # print(sel_p_w)
            sel_p_w_list.append(sel_p_w)
            # print(np.amax(weights))

            # print(weights)
            bias1 = beta
            bias2 = gamma*moving_mean
            bias2 = bias2/(np.sqrt(moving_variance))
            bias = bias1 - bias2
            # print(bias)

            bias, sel_p_b = quantize(bias, sel_p=None)
            sel_p_b_list.append(sel_p_b)
            # print(bias)
            # print(bias.dtype)

            print('############')
            np.set_printoptions(threshold=np.nan)
            print(weights.flatten())
            np.save('/home/nico/semester_project/weights/layer'+str(layer_n)+'_weights', weights)
            np.save('/home/nico/semester_project/weights/layer'+str(layer_n)+'_biases', bias)

        except KeyError:
            print("qua")
            weights = layer_dict['det_q_quantized_weights:0']['values']
            weights, sel_p_w = quantize(weights, sel_p=None)
            print(sel_p_w)
            print(np.amax(weights))

            sel_p_w_list.append(sel_p_w)
            # print(p_weights)
            np.save('/home/nico/semester_project/weights/layer'+str(9)+'_weights', weights)
    np.save('/home/nico/semester_project/weights/sel_p_w', sel_p_w_list)
    np.save('/home/nico/semester_project/weights/sel_p_b', sel_p_b_list)





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