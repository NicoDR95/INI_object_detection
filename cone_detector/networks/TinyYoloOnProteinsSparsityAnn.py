from networks.NetworkBaseSparsityAnn import NetworkBaseSparsityAnn
import tensorflow as tf

class TinyYoloOnProteinsSparsityAnn(NetworkBaseSparsityAnn):

    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__(parameters=parameters)
        if parameters.leaky_relu is True:
            self.activation = self.lrelu
        elif parameters.leaky_relu is False:
            self.activation = self.relu
        else:
            raise ValueError("leaky_relu value is wrong")
        self.sparsity_dict = {}
        self.tensor_size_dict = {}

    def network_build(self, x):

        x = self.conv_layer_bn(x, 16, (3, 3), self.activation, 'convolution_1')
        self.sparsity_dict['l1'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l1'] = self.get_tensor_size(x)

        x = self.conv_layer_bn(x, 32, (3, 3), self.activation, 'convolution_2')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_1')
        self.sparsity_dict['l2'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l2'] = self.get_tensor_size(x)

        x = self.conv_layer_bn(x, 64, (3, 3), self.activation, 'convolution_3')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_2')
        self.sparsity_dict['l3'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l3'] = self.get_tensor_size(x)

        x = self.conv_layer_bn(x, 128, (3, 3), self.activation, 'convolution_4')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_3')
        self.sparsity_dict['l4'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l4'] = self.get_tensor_size(x)

        x = self.conv_layer_bn(x, 256, (3, 3), self.activation, 'convolution_5')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_4')
        self.sparsity_dict['l5'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l5'] = self.get_tensor_size(x)

        x = self.conv_layer_bn(x, 512, (3, 3), self.activation, 'convolution_6')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_5')
        self.sparsity_dict['l6'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l6'] = self.get_tensor_size(x)

        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convolution_7')
        self.sparsity_dict['l7'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l7'] = self.get_tensor_size(x)

        x = self.dropout_layer(x)

        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convolution_8')
        self.sparsity_dict['l8'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l8'] = self.get_tensor_size(x)

        # Last layer without batch normalization and with linear activation
        x = self.detector_layer(x)
        self.sparsity_dict['l9'] = self.get_sparsity_unknown_size(x)
        self.tensor_size_dict['l9'] = self.get_tensor_size(x)

        x = self.reshape_output_layer(x)

        sum_of_sparsities = self.sparsity_dict['l1'] + self.sparsity_dict['l2'] +\
                            self.sparsity_dict['l3'] + self.sparsity_dict['l4'] + self.sparsity_dict['l5'] +\
                            self.sparsity_dict['l6'] + self.sparsity_dict['l7'] + self.sparsity_dict['l8'] +\
                            self.sparsity_dict['l9']
        weighted_sum_of_sparsities = self.sparsity_dict['l1']* self.tensor_size_dict['l1'] + self.sparsity_dict['l2']* self.tensor_size_dict['l2'] + \
                                     self.sparsity_dict['l3']* self.tensor_size_dict['l3'] + self.sparsity_dict['l4']* self.tensor_size_dict['l4'] + self.sparsity_dict['l5']* self.tensor_size_dict['l5'] + \
                                     self.sparsity_dict['l6']* self.tensor_size_dict['l6'] + self.sparsity_dict['l7']* self.tensor_size_dict['l7'] + self.sparsity_dict['l8']* self.tensor_size_dict['l8'] + \
                                     self.sparsity_dict['l9']* self.tensor_size_dict['l9']
        sum_of_sizes = self.tensor_size_dict['l1'] + self.tensor_size_dict['l2'] + \
                       self.tensor_size_dict['l3'] + self.tensor_size_dict['l4'] + self.tensor_size_dict['l5'] + \
                       self.tensor_size_dict['l6'] + self.tensor_size_dict['l7'] + self.tensor_size_dict['l8'] + \
                       self.tensor_size_dict['l9']

        self.sparsity_dict['mean'] = sum_of_sparsities / 9
        self.sparsity_dict['weighted_mean'] = weighted_sum_of_sparsities / sum_of_sizes


        return x, self.sparsity_dict


    def get_sparsity_unknown_size(self, input_tensor):

        tensor_size = tf.cast(tf.size(input_tensor, out_type=tf.int32), dtype=tf.float32)
        return 1.0 - (tf.math.count_nonzero(input_tensor, dtype=tf.float32) / tensor_size)

    def get_tensor_size(self, input_tensor):

        return tf.cast(tf.size(input_tensor, out_type=tf.int32), dtype=tf.float32)
