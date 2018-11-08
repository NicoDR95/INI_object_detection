from networks.NetworkBase import NetworkBase

class FullYolo(NetworkBase):

    def __init__(self, parameters):
        self.parameters = parameters
        super().__init__(parameters=parameters)
        self.activation = self.relu

    def network_build(self, x):
        x = self.conv_layer_bn(x, 32, (3, 3), self.activation, 'convolution1')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool1')
        x = self.conv_layer_bn(x, 64, (3, 3), self.activation, 'convolution2')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool2')

        x = self.conv_layer_bn(x, 128, (3, 3), self.activation, 'convolution3')
        x = self.conv_layer_bn(x, 64, (1, 1), self.activation, 'convolution4')
        x = self.conv_layer_bn(x, 128, (3, 3), self.activation, 'convolution5')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool5')

        x = self.conv_layer_bn(x, 256, (3, 3), self.activation, 'convolution6')
        x = self.conv_layer_bn(x, 128, (1, 1), self.activation, 'convolution7')
        x = self.conv_layer_bn(x, 256, (3, 3), self.activation, 'convolution8')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'maxpool8')

        x = self.conv_layer_bn(x, 512, (3, 3), self.activation, 'convolution9')
        x = self.conv_layer_bn(x, 256, (1, 1), self.activation, 'convolution10')
        x = self.conv_layer_bn(x, 512, (3, 3), self.activation, 'convolution11')
        x = self.conv_layer_bn(x, 256, (1, 1), self.activation, 'convolution12')
        passthrough = self.conv_layer_bn(x, 512, (3, 3), self.activation, 'convolution13')
        x = self.maxpool_layer(passthrough, (2, 2), (2, 2), 'maxpool13')

        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convolution14')
        x = self.conv_layer_bn(x, 512, (1, 1), self.activation, 'convolution15')
        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convolution16')
        x = self.conv_layer_bn(x, 512, (1, 1), self.activation, 'convolution17')
        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convolution18')

        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convolution19')
        x = self.conv_layer_bn(x, 1024, (3, 3), self.activation, 'convoonlution20')

        reordered_layer = self.reorg_layer(passthrough, 'reorg_layer')
        concatenated_layer = self.concat_layers(x, reordered_layer, 'concat_layers')
        x = self.conv_layer_bn(concatenated_layer, 1024, (3, 3), self.activation, 'convlution22')
        x = self.detector_layer(x)

        x = self.reshape_output_layer(x)

        return x
