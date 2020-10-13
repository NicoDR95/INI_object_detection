import math
import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_model_optimization.python.core.quantization.keras.quantizers import Quantizer
from tensorflow_model_optimization.python.core.quantization.keras.quantize_config import QuantizeConfig

import tensorflow_model_optimization.python.core.quantization.keras.quantize as quantize_base

from tensorflow_model_optimization.python.core.quantization.keras.quantize_wrapper import QuantizeWrapper
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_scope
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_apply
from tensorflow_model_optimization.python.core.quantization.keras.quantize import quantize_annotate_layer
from tensorflow_model_optimization.python.core.sparsity.keras.pruning_schedule import PolynomialDecay
from tensorflow.keras.callbacks import Callback
import logging

log = logging.getLogger()



def get_sparsity_fixed_size(input_tensor, num_entries, dtype):
    return 1.0 - (tf.math.count_nonzero(input_tensor, dtype=dtype) / num_entries)


def get_sparsity_unknown_size(input_tensor):
    # fixed use of float32 verified faster than real time size check
    # we use the cast instead of 1 liner because it does work only in eager exeuction
    # 1.0 - (tf.math.count_nonzero(input_tf, dtype=tf.float32) / tf.size(input_tf, out_type=tf.float32)
    tensor_size = tf.cast(tf.size(input_tensor, out_type=tf.int32), dtype=tf.float32)
    return 1.0 - (tf.math.count_nonzero(input_tensor, dtype=tf.float32) / tensor_size)


# base class not to be used
class BaseSparsityMeasurer(Quantizer):

    def __init__(self):
        self.tensor_size_fp = None
        self.tensor_size_fp_dtype = None

    def get_config(self):
        return dict()


    def build(self, tensor_shape, name, layer):
        variable_dict = dict()

        # Activation have batch dimension
        if tensor_shape[0] is not None:
            tensor_size_int = 1
            for dim in tensor_shape:
                tensor_size_int = tensor_size_int * dim

            if tensor_size_int <= tf.float16.max:
                self.tensor_size_fp_dtype = tf.float16
            else:
                self.tensor_size_fp_dtype = tf.float32

            self.tensor_size_fp = tf.dtypes.cast(tensor_size_int, dtype=self.tensor_size_fp_dtype)
        else:
            self.tensor_size_fp = None
            self.tensor_size_fp_dtype = tf.float32

        with tf.device('/:cpu:0'):
            variable_dict["sparsity"] = layer.add_weight(
                name=name + "_sparsity",
                dtype=self.tensor_size_fp_dtype,
                initializer=keras.initializers.Constant(value=0.0),
                trainable=False
            )

        return variable_dict


class WeightsSparsityMeasure(BaseSparsityMeasurer):
    def __call__(self, inputs, training, weights, **kwargs):
        with tf.device('/:cpu:0'):
            sparsity = get_sparsity_fixed_size(inputs, self.tensor_size_fp, self.tensor_size_fp_dtype)
            assign_ops = [weights["sparsity"].assign(sparsity)]

        with tf.control_dependencies(assign_ops):
            inputs = tf.identity(inputs)
            return inputs


class ActivSparsityMeasure(BaseSparsityMeasurer):
    
    def __call__(self, inputs, training, weights, **kwargs):
        if training is False:
            with tf.device('/:cpu:0'):
                sparsity = get_sparsity_unknown_size(tf.nn.relu(inputs))
                sparsity_ops = [weights["sparsity"].assign(sparsity)]
        else:
            sparsity_ops = []

        with tf.control_dependencies(sparsity_ops):
            return tf.identity(inputs)


class SparsityMeter(QuantizeConfig):
    def __init__(self):
        self.weight_measurer = WeightsSparsityMeasure()
        self.output_measurer = ActivSparsityMeasure()

    # Configure how to quantize weights and biases
    def get_weights_and_quantizers(self, layer):

        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            kernel_quant = (layer.depthwise_kernel, self.weight_measurer)
        else:
            try:
                kernel_quant = (layer.kernel, self.weight_measurer)
            except AttributeError:
                log.info("No kernel to report sparsity found for layer {}".format(layer))
                kernel_quant = None

        if kernel_quant is not None:
            return [kernel_quant]
        else:
            return []

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in get_weights_and_quantizers in the same order
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.depthwise_kernel = quantize_weights[0]
        else:
            try:
                layer.kernel = quantize_weights[0]
            except IndexError:
                log.info("No kernel to report sparsity found for layer {}".format(layer))


    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`, in the same order.
        pass

    # Configure how to quantize outputs.
    def get_output_quantizers(self, layer):
        return [self.output_measurer]

    def get_config(self):
        config = dict()
        return config


def measure_sparsity(model):
    assert quantize_base.SET_CUSTOM_TNH_FLAG, log.info("TFMOD needs to be modified with quantizer disabled for proper "
                                                       "running")

    # Helper function uses `quantize_annotate_layer` to annotate that only the
    # Dense layers should be quantized.
    def add_sparsity_annotation(layer):
        quantize_config = SparsityMeter()
        log.info("**Sparsity Measure annotation added to layer {} with {}".format(layer.name, quantize_config))
        quantized_layer = quantize_annotate_layer(to_annotate=layer, quantize_config=quantize_config)
        return quantized_layer

    log.info("Annotating model {}".format(model.name))
    tf.keras.backend.clear_session()
    annotated_model = tf.keras.models.clone_model(model, clone_function=add_sparsity_annotation)

    with quantize_scope({
        'SparsityMeter': SparsityMeter,
        "ActivSparsityMeasure ": ActivSparsityMeasure,
        "WeightsSparsityMeasure ": WeightsSparsityMeasure
    }):
        # Use `quantize_apply` to actually make the model Sparsity Measure aware.
        quant_aware_model = quantize_apply(annotated_model)

        return quant_aware_model
