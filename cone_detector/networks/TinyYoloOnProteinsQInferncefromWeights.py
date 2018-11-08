from networks.NetworkBase import NetworkBase
import numpy as np
import tensorflow as tf

class TinyYoloOnProteinsQuantizedInferencefromWeights(NetworkBase):

    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__(parameters=parameters)
        if parameters.leaky_relu is True:
            self.activation = self.lrelu
        elif parameters.leaky_relu is False:
            self.activation = self.relu
        else:
            raise ValueError("leaky_relu value is wrong")

    def network_build(self, x):

        x = self.inference_layer(x, self.activation, 1, 'conv1')

        x = self.inference_layer(x, self.activation, 2, 'conv2')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool1')

        x = self.inference_layer(x, self.activation, 3, 'conv3')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool2')

        x = self.inference_layer(x, self.activation, 4, 'conv4')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool3')

        x = self.inference_layer(x, self.activation, 5, 'conv5')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool4')

        x = self.inference_layer(x, self.activation, 6, 'conv6')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool5')

        x = self.inference_layer(x, self.activation, 7, 'conv7')

        x = self.inference_layer(x, self.activation, 8, 'conv8')

        # Last layer without batch normalization and with linear activation
        x = self.inference_layer_detector(x, 9, 'conv9')

        x = self.reshape_output_layer(x)

        return x

    def inference_layer(self, x,  activation, layer_n, name):
        variables_path = '/home/nico/semester_project/weights/'
        sel_p_w_list = np.load(variables_path + 'sel_p_w.npy')
        sel_p_w = sel_p_w_list[layer_n - 1]                                   # layer_n starts from 1...
        point_w = -self.parameters.fixed_point_width + sel_p_w
        shift_w = 2**(self.parameters.fixed_point_width - point_w)
        weights = variables_path + 'layer' + str(layer_n) + '_weights.npy'
        weights = np.load(weights)
        weights = weights.astype(dtype='float32')
        weights = weights / shift_w
        weights = tf.Variable(weights, dtype=tf.float32)
        shape_w = weights.shape
        weights = self.quantize_variable(weights, shape_w)

        sel_p_b_list = np.load(variables_path + 'sel_p_b.npy')
        sel_p_b = sel_p_b_list[layer_n - 1]  # layer_n starts from 1...
        point_b = -self.parameters.fixed_point_width + sel_p_b
        shift_b = 2 ** (self.parameters.fixed_point_width - point_b)
        biases = variables_path + 'layer' + str(layer_n) + '_biases.npy'
        biases = np.load(biases)
        biases = biases.astype(dtype='float32')
        biases = biases / shift_b
        biases = tf.Variable(biases, dtype=tf.float32)
        shape_b = biases.shape
        biases = self.quantize_variable(biases, shape_b)

        x = tf.nn.conv2d(
            input=x,
            filter=weights,
            strides=[1, 1, 1, 1],
            padding="SAME",
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=[1, 1, 1, 1],
            name=name
        )

        x = tf.nn.bias_add(x, biases)

        x = activation(x)

        return x

    def inference_layer_detector(self, x, layer_n, name):
        variables_path = '/home/nico/semester_project/weights/'
        sel_p_w_list = np.load(variables_path + 'sel_p_w.npy')
        sel_p_w = sel_p_w_list[layer_n - 1]  # layer_n starts from 1...
        point_w = -self.parameters.fixed_point_width + sel_p_w
        shift_w = 2 ** (self.parameters.fixed_point_width - point_w)
        weights = variables_path + 'layer' + str(layer_n) + '_weights.npy'
        weights = np.load(weights)
        weights = weights.astype(dtype='float32')
        weights = weights / shift_w
        weights = tf.Variable(weights, dtype=tf.float32)
        shape_w = weights.shape
        weights = self.quantize_variable(weights, shape_w)
        # biases = variables_path + 'layer' + str(layer_n) + '_biases.npy'
        # biases = np.load(biases)
        # biases = tf.Variable(biases, dtype=tf.float32)
        # shape = biases.shape
        # biases = self.quantize_variable(biases, shape)

        x = tf.nn.conv2d(
            input=x,
            filter=weights,
            strides=[1, 1, 1, 1],
            padding="SAME",
            use_cudnn_on_gpu=True,
            data_format='NHWC',
            dilations=[1, 1, 1, 1],
            name=name
        )

        # x = tf.nn.bias_add(x, biases)

        # x = tf.nn.relu(x)

        return x