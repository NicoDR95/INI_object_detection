from networks.NetworkBase import NetworkBase
import tensorflow as tf

class TinyYoloOnProteins(NetworkBase):

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

        x = self.conv_layer_bn(x, 16, (3, 3), self.activation, 'convolution_1')

        x = self.conv_layer_bn(x, 32, (3, 3), self.activation, 'convolution_2')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_1')

        x = self.conv_layer_bn(x, 64, (3, 3), self.activation, 'convolution_3')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_2')

        x = self.conv_layer_bn(x, 128, (3, 3), self.activation, 'convolution_4')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_3')

        x = self.conv_layer_bn(x, 256, (3, 3), self.activation, 'convolution_5')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_4')

        x = self.conv_layer_bn(x, 512, (3, 3), self.activation, 'convolution_6')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_5')

        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convolution_7')

        x = self.dropout_layer(x)

        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convolution_8')

        # Last layer without batch normalization and with linear activation
        x = self.detector_layer(x)

        x = self.reshape_output_layer(x)
        return x