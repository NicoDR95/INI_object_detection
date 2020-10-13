from networks.NetworkBase import NetworkBase
import logging
logging.basicConfig(level=logging.DEBUG)
log = logging.getLogger()



class TinyYoloNoPoolQuantized(NetworkBase):

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
        import tensorflow as tf
        x = tf.cast(x, tf.float16)
        log.info("Building TinyYolo network")

        x = self.conv_layer_bn_before_relu_quantized(x, 16, (3, 3), self.activation, 'convolution_1')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_1')                              # 208x208

        x = self.conv_layer_bn_before_relu_quantized(x, 32, (3, 3), self.activation, 'convolution_2')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_2')                              # 104x104

        x = self.conv_layer_bn_before_relu_quantized(x, 64, (3, 3), self.activation, 'convolution_3')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_3')                              # 52x52

        x = self.conv_layer_bn_before_relu_quantized(x, 128, (3, 3), self.activation, 'convolution_4')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_4')                              # 26x26

        x = self.conv_layer_bn_before_relu_quantized(x, 256, (3, 3), self.activation, 'convolution_5')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool_5')                              # 13x13

        x = self.conv_layer_bn_before_relu_quantized(x, 512, (3, 3), self.activation, 'convolution_6')

        x = self.conv_layer_bn_before_relu_quantized(x, 1024, (3, 3), self.activation, 'convolution_7')

        x = self.conv_layer_bn_before_relu_quantized(x, 1024, (3, 3), self.activation, 'convolution_8')

        # Last layer without batch normalization and with linear activation
        x = self.detector_layer_quantized(x)

        x = self.reshape_output_layer(x)
        return x

    # def get_network(self):
    #     self.set_placeholders()
    #     net_output = self.network_build(self.input_ph)
    #
    #     return net_output, self.input_ph, self.train_flag_ph
