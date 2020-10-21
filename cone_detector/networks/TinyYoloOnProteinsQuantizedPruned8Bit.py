from networks.NetworkBase import NetworkBase
from training.Pruning import Pruning


class TinyYoloOnProteinsQuantizedPruned8Bit(NetworkBase):

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
        pruning_config = {
            "initial_sparsity": self.parameters.pruning_initial_sparsity,
            "final_sparsity": self.parameters.pruning_final_sparsity,
            "begin_step": self.parameters.pruning_begin_step,
            "end_step": self.parameters.pruning_end_step,
            "frequency": self.parameters.pruning_incr_frequency
        }

        weight_pruner_conv1 = Pruning(parameters=self.parameters,
                                      tensor_shape=(3, 3, 3, 16),
                                      enable_pruning=self.parameters.enable_pruning,
                                      pruning_config=pruning_config,
                                      layer_name='conv1')
        x = self.conv_layer_bn_before_relu_quantized_pruned(x, 16, (3, 3), self.activation, 8, weight_pruner_conv1, 'conv1')

        weight_pruner_conv2 = Pruning(parameters=self.parameters,
                                      tensor_shape=(3, 3, 16, 32),
                                      enable_pruning=self.parameters.enable_pruning,
                                      pruning_config=pruning_config,
                                      layer_name='conv2')
        x = self.conv_layer_bn_before_relu_quantized_pruned(x, 32, (3, 3), self.activation, 8, weight_pruner_conv2, 'conv2')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool1')

        weight_pruner_conv3 = Pruning(parameters=self.parameters,
                                      tensor_shape=(3, 3, 32, 64),
                                      enable_pruning=self.parameters.enable_pruning,
                                      pruning_config=pruning_config,
                                      layer_name='conv3')
        x = self.conv_layer_bn_before_relu_quantized_pruned(x, 64, (3, 3), self.activation, 8, weight_pruner_conv3, 'conv3')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool2')

        weight_pruner_conv4 = Pruning(parameters=self.parameters,
                                      tensor_shape=(3, 3, 64, 128),
                                      enable_pruning=self.parameters.enable_pruning,
                                      pruning_config=pruning_config,
                                      layer_name='conv4')
        x = self.conv_layer_bn_before_relu_quantized_pruned(x, 128, (3, 3), self.activation, 8, weight_pruner_conv4, 'conv4')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool3')

        weight_pruner_conv5 = Pruning(parameters=self.parameters,
                                      tensor_shape=(3, 3, 128, 256),
                                      enable_pruning=self.parameters.enable_pruning,
                                      pruning_config=pruning_config,
                                      layer_name='conv5')
        x = self.conv_layer_bn_before_relu_quantized_pruned(x, 256, (3, 3), self.activation, 8, weight_pruner_conv5, 'conv5')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool4')

        weight_pruner_conv6 = Pruning(parameters=self.parameters,
                                      tensor_shape=(3, 3, 256, 512),
                                      enable_pruning=self.parameters.enable_pruning,
                                      pruning_config=pruning_config,
                                      layer_name='conv6')
        x = self.conv_layer_bn_before_relu_quantized_pruned(x, 512, (3, 3), self.activation, 8, weight_pruner_conv6, 'conv6')
        x = self.maxpool_layer(x, (2, 2), (2, 2), 'pool5')

        weight_pruner_conv7 = Pruning(parameters=self.parameters,
                                      tensor_shape=(3, 3, 512, 1024),
                                      enable_pruning=self.parameters.enable_pruning,
                                      pruning_config=pruning_config,
                                      layer_name='conv7')
        x = self.conv_layer_bn_before_relu_quantized_pruned(x, 1024, (3, 3), self.activation, 8, weight_pruner_conv7, 'conv7')

        weight_pruner_conv8 = Pruning(parameters=self.parameters,
                                      tensor_shape=(3, 3, 1024, 1024),
                                      enable_pruning=self.parameters.enable_pruning,
                                      pruning_config=pruning_config,
                                      layer_name='conv8')
        x = self.conv_layer_bn_before_relu_quantized_pruned(x, 1024, (3, 3), self.activation, 8, weight_pruner_conv8, 'conv8')

        # Last layer without batch normalization and with linear activation
        x = self.detector_layer_quantized(x, 8)

        x = self.reshape_output_layer(x)

        return x