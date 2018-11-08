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
        self.input_ph = tf.placeholder(shape=[None, self.parameters.input_h, self.parameters.input_w, self.parameters.input_depth], dtype=tf.float32,
                                       name='image_placeholder')

        self.train_flag_ph = tf.placeholder(dtype=tf.bool, name='flag_placeholder')

    def get_network(self):
        self.set_placeholders()
        net_output = self.network_build(self.input_ph)

        return net_output, self.input_ph, self.train_flag_ph
