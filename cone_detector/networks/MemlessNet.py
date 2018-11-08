import tensorflow as tf

from networks.NetworkBase import NetworkBase


class MemlessNet(NetworkBase):

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


        with tf.name_scope("MemlessNet"):

            x = self.conv_layer_bn_before_relu(x, 16, (3, 3), self.activation, 'conv1')

            x = self.conv_layer_bn_before_relu(x, 32, (3, 3), self.activation, 'conv2')
            x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool1')

            x = self.conv_layer_bn_before_relu(x, 64, (3, 3), self.activation, 'conv3')
            x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool2')

            x = self.conv_layer_bn_before_relu(x, 128, (3, 3), self.activation, 'conv4')
            x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool3')

            x = self.conv_layer_bn_before_relu(x, 256, (3, 3), self.activation, 'conv5')
            x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool4')

            x = self.conv_layer_bn_before_relu(x, 512, (3, 3), self.activation, 'conv6')
            x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool5')

            x = self.conv_layer_bn_before_relu(x, 1024, (3, 3), self.activation, 'conv7')

            x = self.conv_layer_bn_before_relu(x, 1024, (3, 3), self.activation, 'conv8')

        # Last layer without batch normalization and with linear activation

        x = self.detector_layer(x)
        x = self.reshape_output_layer(x)

        return x
