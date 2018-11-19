import logging

import numpy as np
import tensorflow as tf

log = logging.getLogger()
import tensorflow.contrib.slim as slim
import time
from functools import reduce
import operator
import math


class NetworkBase(object):
    n_coordinates = 4
    n_confidences = 1

    def __init__(self, parameters):
        self.parameters = parameters
        self.dropout_idx = 0
        self.n_output_values_per_box = self.n_coordinates + self.n_confidences + self.parameters.n_classes

    def dropout_layer(self, x):
        dropout_value = self.parameters.dropout[self.dropout_idx]
        self.dropout_idx += 1
        x = tf.layers.dropout(inputs=x, rate=dropout_value, training=self.train_flag_ph)
        return x

    def dropout_layer_inference(self, x):
        self.dropout_idx += 1
        x = tf.layers.dropout(inputs=x, rate=0.0, training=self.train_flag_ph)
        return x

    def maxpool_layer(self, x, size, stride, name):
        with tf.name_scope(name):
            x = tf.layers.max_pooling2d(x, size, stride, padding='SAME')

        return x

    def quantize(self, weights, shift, max_value, name, use_float16, use_bfloat16):

        quantizing = weights * shift

        omap = {'Round': 'Identity'}
        # Round doesnt have a gradient, we force it to identity
        with tf.get_default_graph().gradient_override_map(omap):
            quantizing = tf.round(quantizing, name=name + "_round")

        quantizing = tf.clip_by_value(quantizing, -max_value, max_value - 1, name=name + "_clip")
        quantizing = quantizing / shift

        if use_float16:
            quantizing = tf.cast(quantizing, tf.float16)
        elif use_bfloat16:
            quantizing = tf.cast(quantizing, tf.bfloat16)

        return quantizing

    def get_overflow_rate(self, to_quantize, max_value):
        mask = tf.logical_or(
            to_quantize > max_value,
            to_quantize <= -max_value
        )

        overflow_rate = tf.reduce_sum(tf.to_int32(mask))

        return overflow_rate

    # Returns true if there is no overflow
    def get_if_not_overflow(self, max_array, min_array, max_value):

        max_overflow = max_array < max_value
        min_overflow = min_array >= -max_value

        return tf.logical_and(max_overflow, min_overflow)

    def get_quantized_kernel(self, shape, name):
        initializer = tf.contrib.layers.xavier_initializer()

        with tf.device("/cpu:0"):
            weights = tf.Variable(initial_value=initializer(shape=shape),
                                  trainable=True,
                                  collections=None,
                                  validate_shape=True,
                                  caching_device=None,
                                  name=name + "_quantized_weights",
                                  variable_def=None,
                                  dtype=tf.float32
                                  )
        return self.quantize_variable(weights, shape, width=self.parameters.fixed_point_width_weights)

    # TODO add to params
    # TODO CHANGE TO 16 NICOLO
    def quantize_variable(self, variable, shape, width):
        name = variable.name[:-2]

        num_entries = reduce(lambda x, y: x * y, shape)

        use_float16 = True
        use_bfloat16 = False
        '''if 8 <= width <= 10:
            use_float16 = True
            use_bfloat16 = False
        elif 1 <= width <= 7:
            use_float16 = False
            use_bfloat16 = True
        '''
        overflow_threshold_input = 0.0
        # smarted than dividing the whole array by the num of weights
        # Working with int to minimze impact of computation
        rates = []
        shifts = []
        shift_python = {}
        max_value_clip = {}
        max_value = 2.0 ** (width - 1)

        p_start = -width
        p_end = width + 1
        while_loop_iter = p_end - p_start

        with tf.device("/cpu:0"):
            # Prepare constants
            true_fn = tf.constant(False, dtype=tf.bool)
            false_fn = tf.constant(True, dtype=tf.bool)

            for point in range(p_start, p_end - 1):
                shift_python[point] = (2.0 ** (width - point))
                max_value_clip[point] = max_value / shift_python[point]
                shift_tf = tf.constant(shift_python[point], dtype=variable.dtype)
                shifts.append(shift_tf)

            shifts.append(tf.constant(2.0 ** (width - p_end - 1), dtype=variable.dtype))
            shifts = tf.stack(shifts)

            if overflow_threshold_input == 0.0:

                max_array = tf.reduce_max(variable)
                min_array = tf.reduce_min(variable)

                for point in range(p_start, p_end - 1):
                    rate = self.get_if_not_overflow(max_array, min_array, max_value_clip[point])
                    rates.append(rate)

            else:

                for point in range(p_start, p_end - 1):
                    rate = self.get_overflow_rate(variable, max_value_clip[point])
                    rates.append(rate)

            rates = tf.stack(rates)

            if overflow_threshold_input == 0.0:
                rate_thresholded = rates
            else:
                overflow_threshold = tf.constant(int(overflow_threshold_input * num_entries))
                rate_thresholded = rates <= overflow_threshold

            # Highest p is not computed and forced to valid to save ops and ensure operations
            rate_thresholded = tf.concat([rate_thresholded, [True]], axis=0)  # For most significant bit

            def loop_cond(loop_p):
                return tf.cond(
                    # pred=tf.logical_and(rates[selected_p] <= overflow_threshold, selected_p < width),
                    pred=rate_thresholded[loop_p],
                    true_fn=lambda: true_fn,
                    false_fn=lambda: false_fn

                )

            def loop_body(loop_p):
                loop_p = loop_p + 1
                return loop_p

            selected_p = tf.while_loop(
                cond=loop_cond,
                body=loop_body,
                loop_vars=[0],
                parallel_iterations=while_loop_iter,
                # maximum_iterations=while_loop_iter,
                back_prop=False,
                swap_memory=True

            )

            quant_shift = shifts[selected_p]

        quantized_variable = self.quantize(variable, quant_shift, max_value, name, use_float16, use_bfloat16)

        # TODO debug
        if name is "conv1" or name is "det_q":
            with tf.device("/cpu:0"):
                quantized_variable = tf.Print(quantized_variable, [selected_p, tf.reduce_max(variable), tf.reduce_max(quantized_variable)],
                                              name + " selected_p, max unquantized weight, max quantized weight: ")

        return quantized_variable

    def get_quantized_conv(self, x, out_ch, kernel, name, add_biases=False):
        kernel_shape = list(kernel) + [int(x.shape[3]), out_ch]
        kernels = self.get_quantized_kernel(kernel_shape, name)

        x = tf.nn.conv2d(
            input=x,
            filter=kernels,
            strides=[1, 1, 1, 1],
            padding="SAME",
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=[1, 1, 1, 1],
            name=name
        )

        shape = [self.parameters.batch_size] + x.get_shape().as_list()[1:]

        if add_biases:
            biases = tf.Variable(-1 * np.ones(shape=(out_ch,)), dtype=tf.float32)
            biases = self.quantize_variable(biases, (out_ch,), width=self.parameters.fixed_point_width_weights)
            x = tf.nn.bias_add(x, biases)

        x = self.quantize_variable(x, shape, width=self.parameters.fixed_point_width_activation)
        return x

    def conv_layer_bn_before_relu_quantized(self, x, out_ch, kernel, activation_func, name):
        x = self.get_quantized_conv(x, out_ch, kernel, name)

        x = tf.layers.batch_normalization(inputs=x, training=self.train_flag_ph, momentum=0.99, epsilon=0.001, center=True,
                                          scale=True, name=name + '_bn')

        x = activation_func(x)

        return x

    def detector_layer_quantized(self, x):
        # Last layer without batch normalization and with linear activation (None argument)
        x = self.get_quantized_conv(x=x, out_ch=self.parameters.n_anchors * self.n_output_values_per_box, kernel=(1, 1),
                                    name='det_q', add_biases=True)

        return x

    def conv_layer_bn_before_relu(self, x, depth, kernel, activation_func, name):
        x = slim.conv2d(inputs=x, num_outputs=depth, kernel_size=kernel, activation_fn=activation_func,
                        normalizer_fn=slim.batch_norm, normalizer_params={'is_training': self.train_flag_ph}, scope=name)
        return x

    def conv_layer_bn(self, x, depth, kernel, activation_func, name):
        x = slim.conv2d(inputs=x, num_outputs=depth, kernel_size=kernel, activation_fn=activation_func,
                        normalizer_fn=slim.batch_norm, normalizer_params={'is_training': self.train_flag_ph},
                        scope=name)

        return x

    def conv_layer_bn_inference(self, x, depth, kernel, activation_func, name):
        x = tf.layers.conv2d(inputs=x, filters=depth, kernel_size=kernel, padding='SAME', activation=activation_func,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             bias_initializer=tf.zeros_initializer(), name=name)
        x = tf.layers.batch_normalization(inputs=x, training=False, momentum=0.99, epsilon=0.001, center=True,
                                          scale=True, name=name + '_bn')
        return x

    def conv_layer(self, x, depth, kernel, activation_func, name):
        x = tf.layers.conv2d(inputs=x, filters=depth, kernel_size=kernel, padding='SAME', activation=activation_func,
                             kernel_initializer=tf.contrib.layers.xavier_initializer_conv2d(),
                             bias_initializer=tf.zeros_initializer(), name=name)
        return x

    def reorg_layer(self, x, name):
        patch = 2
        stride = 2
        x = tf.extract_image_patches(images=x, ksizes=[1, patch, patch, 1],
                                     strides=[1, stride, stride, 1],
                                     rates=[1, 1, 1, 1], padding='VALID',
                                     name=name)
        return x

    def concat_layers(self, x, reordered_layer, name):
        x = tf.concat(values=[x, reordered_layer], axis=3, name=name)
        return x

    def detector_layer(self, x):
        # Last layer without batch normalization and with linear activation (None argument)
        x = self.conv_layer(x=x, depth=self.parameters.n_anchors * self.n_output_values_per_box, kernel=(1, 1), activation_func=None,
                            name='detector_layer')
        return x

    def reshape_output_layer(self, x):
        x = tf.reshape(tensor=x,
                       shape=(-1, self.parameters.output_h, self.parameters.output_w, self.parameters.n_anchors, self.n_output_values_per_box),
                       name='network_output')
        return x

    def relu(self, x):
        return tf.nn.relu(x, name="relu")

    def lrelu(self, x):
        return tf.nn.leaky_relu(x, name="relu")

    def set_placeholders(self):
        # with tf.device("/gpu:0"):
        # Placeholder for input batch of images
        log.info("Creating input placeholder...")
        self.input_ph = tf.placeholder(shape=[None, self.parameters.input_h, self.parameters.input_w, self.parameters.input_depth],
                                       dtype=tf.float32,
                                       name='image_placeholder')

        self.train_flag_ph = tf.placeholder(dtype=tf.bool, name='flag_placeholder')

    def get_network(self):
        self.set_placeholders()
        net_output = self.network_build(self.input_ph)

        return net_output, self.input_ph, self.train_flag_ph

    '''
    ***********
    EXPERIMENTAL MEMLESS FUNCTIONS
    ***********
    '''

    def get_memless_gen_layer(self, x, num_units, name, special_nonlinearity):
        num_input_units = int(x.shape[-1])
        layer_w = self.get_memless_var(name + "_lw", [num_input_units, num_units])
        layer_b = self.get_memless_var(name + "_lb", [num_units], zero_init=True)
        layer_out = tf.nn.bias_add(tf.matmul(x, layer_w, name=name + "_matmul"), layer_b, name=name + "_bias_add")

        if special_nonlinearity is False:
            # layer_out = tf.nn.leaky_relu(layer_out, (1.0 / (2 ** 10)), name=name + "_leakyrelu")
            # layer_out = tf.nn.sigmoid(layer_out)
            layer_out = tf.nn.selu(layer_out)
            # layer_out = self.sqrtsin_nonlinearity(pre_activ)
        else:
            # layer_out = tf.sin(59*layer_out)
            # for prime in reversed([41, 37, 31]): #41 37
            #    layer_out = tf.sin(prime * layer_out) + layer_out
            layer_out = tf.sin(3 * layer_out)
            for prime in reversed([7, 13, 2]):  # 41 37
                layer_out = tf.sin(prime * layer_out) + layer_out

            # layer_out = tf.nn.relu(layer_out, name=name + "_relu")
            # layer_out = self.sqrtsin_nonlinearity(layer_out)
            # layer_out = pre_activ

        return layer_out

    def sqrtsin_nonlinearity(self, x):
        r = tf.sqrt(-2.0 * tf.log(x))
        sigma = 2.0 * math.pi * x
        nonlinear = r * tf.cos(sigma)
        return nonlinear

    def test_nonlinearity(self, x):
        x = x * tf.sin(x) + x
        return x

    def log_kernel(self, kernel):

        with tf.device("/cpu:0"):
            num_positives = tf.count_nonzero(tf.greater_equal(kernel, 0.))
            num_negatives = tf.size(kernel) - tf.to_int32(num_positives)
            kernel = tf.Print(kernel, [num_positives], "num_positives:", first_n=-1, summarize=128)
            kernel = tf.Print(kernel, [num_negatives], "num_negatives:", first_n=-1, summarize=128)
            kernel = tf.Print(kernel, [kernel[:, :, 0, 0]], "Kernel[:,:,0,0]: ", first_n=-1, summarize=128)
            kernel = tf.Print(kernel, [kernel[:, :, 0, 1]], "Kernel[:,:,0,1]: ", first_n=-1, summarize=128)

            kernel = tf.Print(kernel, [kernel[:, :, 1, 0]], "Kernel[:,:,1,0]: ", first_n=-1, summarize=128)
            kernel = tf.Print(kernel, [kernel[:, :, 1, 1]], "Kernel[:,:,1,1]: ", first_n=-1, summarize=128)

        return kernel

    def mult_add(self, x, mult, add, name):
        ret = tf.nn.bias_add(tf.matmul(x, mult, name=name + "_matmul"), add, name=name + "_bias")
        return ret

    def mult_add_leaky(self, x, mult, add, leak, name):
        matmult = self.mult_add(x, mult, add, name)
        ret = tf.nn.leaky_relu(matmult, leak, name=name + "_nl")
        return ret

    def check_var(self, var):
        with tf.device("/cpu:0"):
            var = tf.Print(var, [var], "check var", first_n=1, summarize=128)
        return var

    def get_memless_kernel(self, kernel_shape, name):

        start = time.time()
        global_start = start
        name_und = name + "_"
        scope_name = name_und + "memless_network"

        expansion_num_units = 128
        extraction_num_units = 18

        leaky_coeff = 1.0 / 256.0

        num_out_ch = kernel_shape[3]
        shape_out_first = [kernel_shape[3], kernel_shape[0], kernel_shape[1], kernel_shape[2]]
        num_weight_per_out_ch = kernel_shape[0] * kernel_shape[1] * kernel_shape[2]
        num_extract_steps = int(math.ceil(kernel_shape[0] * kernel_shape[1] * kernel_shape[2] / extraction_num_units))

        log.info("Num extract steps: {}".format(num_extract_steps))

        with tf.name_scope(scope_name) and tf.variable_scope(scope_name):
            # Seeds specific for each out ch
            seeds = self.get_memless_var(name=name_und + "seeds", shape=[num_out_ch, expansion_num_units], random_uniform=True)

            # extract stage. Here we need to get the actual weights. The input is of shape [num_out_ch, dim_num_units]
            expansion_m = self.get_memless_var(name=name_und + "expansion_m", shape=[expansion_num_units, expansion_num_units])
            expansion_b = self.get_memless_var(name=name_und + "expansion_b", shape=[expansion_num_units], zero_init=True)

            extraction_m = self.get_memless_var(name=name_und + "extraction_m", shape=[expansion_num_units, extraction_num_units])
            extraction_b = self.get_memless_var(name=name_und + "extraction_b", shape=[extraction_num_units], zero_init=True)

            expanded_data = 0.0

            extracted_data_list = list()
            for extract_step in range(num_extract_steps):
                expand_basename = name_und + "expand_{}_".format(extract_step)
                extract_basename = name_und + "extract_{}_".format(extract_step)

                expanded_data = expanded_data + seeds
                expanded_data = self.mult_add_leaky(expanded_data, expansion_m, expansion_b, leaky_coeff, expand_basename)
                # Result appended before non linearity
                extracted_data = self.mult_add(expanded_data, extraction_m, extraction_b, extract_basename)

                extracted_data_list.append(extracted_data)

            concat_weights = tf.concat(extracted_data_list, axis=1)
            # print(concat_weights.shape)
            cut_unused = tf.slice(concat_weights, begin=[0, 0], size=[num_out_ch, num_weight_per_out_ch])
            rescale_m = self.get_memless_var(name=name_und + "rescale_m", shape=[1], value_init=1.0)
            rescale_b = self.get_memless_var(name=name_und + "rescale_b", shape=[1], zero_init=True)
            cut_unused = cut_unused * rescale_m + rescale_b

            # print(cut_unused.shape)
            reshaped_out_first = tf.reshape(cut_unused, shape_out_first)
            kernel = tf.transpose(reshaped_out_first, [1, 2, 3, 0])

            # kernel = self.log_kernel(kernel)

        # sanity check on shape
        for shape_index in range(4):
            assert (int(kernel.shape[shape_index]) == kernel_shape[shape_index])

        # LOGs
        all_variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope_name)

        num_variables = 0
        for var in all_variables:
            var_shape = var.shape
            var_shape_list = [var_shape[i].value for i in range(len(var_shape))]
            num_var = reduce(operator.mul, var_shape_list, 1)
            num_variables = num_variables + num_var

        original_num_weights = reduce(operator.mul, kernel_shape, 1)
        compr_ratio = 100.0 * num_variables / original_num_weights
        timelapse = time.time() - global_start
        log.info("Created kernel {} ({}). Num effective variables: {} ({:2f}%) ({:2f}s)".format(name, kernel.shape, num_variables, compr_ratio,
                                                                                                timelapse))
        return kernel

    def get_memless_var(self, name, shape, zero_init=False, random_uniform=False, normal_init=False, value_init=None):
        var_name = name + "_memless"
        regularizer = None
        # regularizer = tf.contrib.layers.l2_regularizer(1.0)
        # regularizer = tf.contrib.layers.l1_regularizer(1.0)
        # regularizer = tf.contrib.layers.l1_l2_regularizer(0.1,0.1)

        if zero_init is True:
            initializer = tf.zeros_initializer()
        elif random_uniform is True:
            initializer = tf.initializers.random_uniform(minval=-1, maxval=1)
        elif normal_init is True:
            initializer = tf.initializers.random_normal()
        elif value_init is not None:
            initializer = value_init
            shape = None
        else:
            initializer = tf.contrib.layers.xavier_initializer()
            # initializer = tf.zeros_initializer()
            # initializer = tf.initializers.random_normal()

        var = tf.get_variable(
            name=var_name,
            shape=shape,
            dtype=tf.float32,
            initializer=initializer,
            regularizer=regularizer,
            trainable=True,  ###TODO
            validate_shape=True,
        )

        with tf.device("/cpu:0"):
            var = tf.Print(var, [var], var_name, first_n=1, summarize=36)
        return var

    def get_memless_conv(self, x, out_ch, kernel, name, add_biases=False):
        kernel_shape = list(kernel) + [int(x.shape[3]), out_ch]
        kernels = self.get_memless_kernel(kernel_shape, name)

        x = tf.nn.conv2d(
            input=x,
            filter=kernels,
            strides=[1, 1, 1, 1],
            padding="SAME",
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=[1, 1, 1, 1],
            name=name
        )

        if add_biases:  # todo param shape
            biases = tf.Variable(-5 * np.ones(shape=(out_ch,)), dtype=tf.float32)
            biases = self.quantize_variable(biases, (out_ch,), width=16)
            x = tf.nn.bias_add(x, biases)

        return x

    def conv_layer_bn_before_relu_memless(self, x, out_ch, kernel, activation_func, name):
        x = self.get_memless_conv(x, out_ch, kernel, name)

        x = tf.layers.batch_normalization(inputs=x, training=self.train_flag_ph, momentum=0.99, epsilon=0.001, center=True,
                                          scale=True, name=name + '_bn')

        x = activation_func(x)

        return x
