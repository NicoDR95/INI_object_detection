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

_WEIGHTS_NUM_BITS = 8
_ACTIV_NUM_BITS = 16
float_policy = tf.float32

quantization_map = [
    tf.keras.layers.Dense,
    tf.keras.layers.Conv2D,
    tf.keras.layers.Conv2DTranspose
    # tf.keras.layers.Input: BFPInputQuantizerConfig()
]


def get_quant_exp(variable, num_bits, use_abs, return_abs):
    if use_abs:
        abs_variable = tf.abs(variable)
    else:
        abs_variable = variable

    max_variable = tf.reduce_max(abs_variable)

    # we need a conditional statement because of the array is entirely 0 we would get a -inf log
    log_variable = tf.cond(
        tf.equal(max_variable, 0.0),
        lambda: tf.constant(0.0, variable.dtype),
        lambda: tf.math.log(max_variable) / tf.constant(math.log(2.0), variable.dtype)
        # tf doesnt have log2, so we compute log in base e and then divide to cast it to log2
    )

    log_rounded = tf.cast(tf.math.ceil(log_variable), tf.int16)
    unsigned_width = num_bits - 1  # for the sign bit
    exp_diff = unsigned_width - log_rounded
    no_grad_exp = tf.stop_gradient(exp_diff)
    if return_abs is False:
        return no_grad_exp
    else:
        return no_grad_exp, abs_variable


def quantize_variable(variable, exp, width, clip, clip_negative=False, round_or_floor="round", name=""):
    # Use tf.custom_gradient to add a gradient to round and floor operations
    # We need a sub function because @tf.custom_gradient doesnt support kwargs
    @tf.custom_gradient
    def _quantize_variable(l_var):
        def grad(dy):  # identity gradient
            # if clip_negative is False:
            return dy
            # else:
            #     raise ("Negative clipping not implemented")
            #     var_mask = tf.cast(tf.greater_equal(variable, 0.0), variable.dtype)
            #     dy = tf.multiply(variable, var_mask)
            #     return dy

        # Round doesnt have a gradient, we force it to identity
        # quant_shift = 2.0 ** tf.cast(exp, tf.float32)  # cast for ** compatiblity
        quant_shift = 2.0 ** tf.cast(exp, l_var.dtype)  # cast for ** compatiblity
        quantizing = l_var * quant_shift

        if round_or_floor == "round":
            quantizing = tf.round(quantizing, name=name + "_round")
        elif round_or_floor == "floor":
            quantizing = tf.floor(quantizing, name=name + "_floor")
        else:
            raise ValueError("Illegal round mode {}".format(round_or_floor))

        if clip is True or clip_negative is True:
            max_value = 2.0 ** (width - 1)
            if clip_negative is True:
                min_value = 0
            else:
                min_value = -max_value

            quantizing = tf.clip_by_value(quantizing, min_value, max_value - 1, name=name + "_clip")

        quantizing = quantizing / quant_shift
        return quantizing, grad

    quantized_variable = _quantize_variable(variable)
    return quantized_variable


def get_sparsity_fixed_size(input_tensor, num_entries, dtype):
    return 1.0 - (tf.math.count_nonzero(input_tensor, dtype=dtype) / num_entries)


def get_sparsity_unknown_size(input_tensor):
    # fixed use of float32 verified faster than real time size check
    # we use the cast instead of 1 liner because it does work only in eager exeuction
    # 1.0 - (tf.math.count_nonzero(input_tf, dtype=tf.float32) / tf.size(input_tf, out_type=tf.float32)
    tensor_size = tf.cast(tf.size(input_tensor, out_type=tf.int32), dtype=tf.float32)
    return 1.0 - (tf.math.count_nonzero(input_tensor, dtype=tf.float32) / tensor_size)


# base class not to be used
class BFPQuantizer(Quantizer):

    def __init__(self, num_bits):
        self.num_bits = num_bits
        self.tensor_size_int = None
        self.tensor_size_fp = None
        self.tensor_size_fp_dtype = None

    def get_config(self):
        config = dict()
        config["num_bits"] = self.num_bits
        return config

    def build(self, tensor_shape, name, layer):
        variable_dict = dict()

        # Activation have batch dimension
        if tensor_shape[0] is not None:
            self.tensor_size_int = 1
            for dim in tensor_shape:
                self.tensor_size_int = self.tensor_size_int * dim

            if self.tensor_size_int <= tf.float16.max:
                self.tensor_size_fp_dtype = tf.float16
            else:
                self.tensor_size_fp_dtype = tf.float32

            self.tensor_size_fp = tf.dtypes.cast(self.tensor_size_int, dtype=self.tensor_size_fp_dtype)
        else:
            self.tensor_size_int = None
            self.tensor_size_fp = None
            self.tensor_size_fp_dtype = tf.float32

        variable_dict["exp"] = layer.add_weight(
            name=name + "_exp",
            dtype=tf.int16,
            initializer=keras.initializers.Constant(value=self.num_bits),
            # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
            trainable=False
        )

        with tf.device('/:cpu:0'):
            variable_dict["sparsity"] = layer.add_weight(
                name=name + "_sparsity",
                dtype=self.tensor_size_fp_dtype,
                initializer=keras.initializers.Constant(value=0.0),
                trainable=False
            )

        return variable_dict


class BFPWeightQuantizer(BFPQuantizer):
    # Pruning type furnished either as class type and not an object
    # Pruning config furnished as dictionary
    def __init__(self, num_bits, enable_pruning=False, pruning_schedule=None):
        super().__init__(num_bits)

        self.enable_pruning = enable_pruning
        self.pruning_schedule = pruning_schedule

    def get_config(self):
        config = super().get_config()
        config["pruning_schedule"] = tf.keras.utils.serialize_keras_object(self.pruning_schedule)
        config["enable_pruning"] = self.enable_pruning
        return config

    @classmethod
    def from_config(cls, config):
        try:
            # workaround to deal with some pruning schedule classes not bein serialized properly
            del config["pruning_schedule"]["class_name"]
            config["pruning_schedule"] = tf.keras.utils.deserialize_keras_object(config["pruning_schedule"]["config"])
        except:
            pass
        return cls(**config)

    def build(self, tensor_shape, name, layer):
        self.name = layer.name
        variable_dict = super().build(tensor_shape, name, layer)

        variable_dict["stored_tensor"] = layer.add_weight(
            name=name + "_stored_tensor",
            shape=tensor_shape,
            initializer=keras.initializers.glorot_normal(),
            # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
            trainable=False,
            dtype=float_policy
        )

        if self.enable_pruning is True:
            # Pruning variables
            variable_dict["mask"] = layer.add_weight(
                'mask',
                shape=tensor_shape,
                initializer=tf.keras.initializers.get('zeros'),
                dtype=variable_dict["stored_tensor"].dtype,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN)

            variable_dict["threshold"] = layer.add_weight(
                'threshold',
                shape=[],
                initializer=tf.keras.initializers.get('ones'),
                dtype=variable_dict["stored_tensor"].dtype,
                trainable=False,
                aggregation=tf.VariableAggregation.MEAN)

            variable_dict["pruning_step"] = layer.add_weight(
                'pruning_step',
                shape=[],
                initializer=tf.keras.initializers.Constant(-1),
                dtype=tf.int64,
                trainable=False)

        return variable_dict

    def __call__(self, inputs, training, weights, **kwargs):
        # tf.print(self.name + "\n")
        # Weights mode and we aren't in training
        # don't need to be re quantized outside training
        if training is False:
            return weights["stored_tensor"]
        else:
            assign_ops = []

            if self.enable_pruning is True:
                # Get quantization exp and absoluted array
                quant_exp, abs_weights = get_quant_exp(inputs, self.num_bits, use_abs=True, return_abs=True)

                def update_mask(sparsity):
                    # Compute position of threshold in the array.r when sparsity is  Must be capped to avoid erro0
                    density = tf.cast((1 - sparsity), self.tensor_size_fp_dtype)
                    pruning_threshold_pre_round = self.tensor_size_fp * density

                    pruning_threshold_uncapped = tf.math.round(pruning_threshold_pre_round)
                    pruning_threshold_index_uncapped = tf.dtypes.cast(pruning_threshold_uncapped, tf.int32)

                    pruning_threshold_index = tf.math.minimum(pruning_threshold_index_uncapped,
                                                              self.tensor_size_int - 1)

                    # Sort the entire array (flattened)
                    sorted_weights, _ = tf.math.top_k(tf.reshape(abs_weights, [-1]), k=self.tensor_size_int)

                    # Selected the threshold value
                    current_threshold = sorted_weights[pruning_threshold_index]

                    # compare to get new mask
                    mask_bool = tf.math.greater(abs_weights, current_threshold)
                    mask = tf.dtypes.cast(mask_bool, inputs.dtype)

                    assign_ops.append(weights["mask"].assign(mask))
                    assign_ops.append(weights["threshold"].assign(current_threshold))

                    return mask

                # Get new sparsity level
                update_pruning, sparsity = self.pruning_schedule(weights["pruning_step"])
                sel_mask = tf.cond(update_pruning, lambda: update_mask(sparsity), lambda: weights["mask"])
                no_grad_mask = tf.stop_gradient(sel_mask)
                masked_weight = tf.math.multiply(inputs, no_grad_mask)
                incremented_pruning_step = weights["pruning_step"] + 1
                # updated even if not pruning since we need it for keeping track of when to prune
                assign_ops.append(weights["pruning_step"].assign(incremented_pruning_step))

            else:
                # Get quantization exp and absoluted array
                quant_exp = get_quant_exp(inputs, self.num_bits, use_abs=True, return_abs=False)
                masked_weight = inputs

            # Quantization
            quantized_inputs = quantize_variable(variable=masked_weight,
                                                 exp=quant_exp,
                                                 width=self.num_bits,
                                                 clip=False,
                                                 round_or_floor="round",
                                                 name="weight")

            sparsity = get_sparsity_fixed_size(quantized_inputs, self.tensor_size_fp, self.tensor_size_fp_dtype)

            assign_ops.append(weights["sparsity"].assign(sparsity))
            assign_ops.append(weights["stored_tensor"].assign(quantized_inputs))
            assign_ops.append(weights["exp"].assign(quant_exp))

            with tf.control_dependencies(assign_ops):
                quantized_inputs = tf.identity(quantized_inputs)

            return quantized_inputs


class BFPActivQuantizer(BFPQuantizer):
    def __init__(self, num_bits, num_batch):
        super().__init__(num_bits)
        self.num_batch = num_batch

    def get_config(self):
        config = super().get_config()
        config["num_batch"] = self.num_batch
        return config

    def build(self, tensor_shape, name, layer):
        variable_dict = super().build(tensor_shape, name, layer)
        with tf.device('/:cpu:0'):
            variable_dict["exp_memory"] = layer.add_weight(
                name=name + "_exp_memory",
                dtype=tf.int16,
                shape=(self.num_batch,),
                initializer=keras.initializers.Constant(value=0),
                trainable=False
            )

            variable_dict["exp_memory_ptr"] = layer.add_weight(
                name=name + "_exp_memory_ptr",
                dtype=tf.int32,
                initializer=keras.initializers.Constant(value=0),
                trainable=False
            )




            if len(tensor_shape) == 4:
                store_shape = [tensor_shape[1], tensor_shape[2], tensor_shape[3]]
            else:
                store_shape = [tensor_shape[1], ]

            variable_dict["stored_tensor"] = layer.add_weight(
                name=name + "_stored_tensor",
                shape=store_shape,
                initializer=None,
                # assume that the max value of the variable is 1, so we put it to output_num_bits and minimize loss
                trainable=False,
                dtype=float_policy
            )

            use_bias_layer = layer
            while isinstance(use_bias_layer, QuantizeWrapper):
                use_bias_layer = layer.layer

            self.use_bias = use_bias_layer.use_bias
            if use_bias_layer.use_bias:
                for var_idx, var in enumerate(layer.non_trainable_variables):
                    if "bias_exp" in var.name:
                        variable_dict["output_to_bias_exp"] = var  # very weak, but how to get the right one?
                        break
                else:
                    raise AttributeError("No bias exponent found for quantization for tensor {}".format(name))

        return variable_dict

    def __call__(self, inputs, training, weights, **kwargs):

        if training is True:
            with tf.device('/:cpu:0'):
                new_exp = get_quant_exp(inputs, self.num_bits, use_abs=False, return_abs=False)
                new_memory_ptr = tf.math.floormod(weights["exp_memory_ptr"] + 1, self.num_batch)

                exp_memory_assign_op = weights["exp_memory"][weights["exp_memory_ptr"]].assign(new_exp)
                exp_memory_ptr_assign_op = weights["exp_memory_ptr"].assign(new_memory_ptr)

                quant_exp = tf.reduce_max(weights["exp_memory"]) - 1  # -1 for avoiding overflow
                exp_assign_op = weights["exp"].assign(quant_exp)
                clip = False
                control_ops = [exp_memory_assign_op, exp_memory_ptr_assign_op, exp_memory_ptr_assign_op, exp_assign_op
                               ]
                if self.use_bias:
                    output_to_bias_exp_update = weights["output_to_bias_exp"].assign(quant_exp)
                    control_ops.append(output_to_bias_exp_update)



        else:
            clip = True
            control_ops = []

        quant_exp = weights["exp"]

        with tf.control_dependencies(control_ops):
            quantized_inputs = quantize_variable(variable=inputs,
                                                 exp=quant_exp,
                                                 width=self.num_bits,
                                                 clip=clip,
                                                 clip_negative=False,
                                                 round_or_floor="floor",
                                                 name="activ")

            if training is False:
                with tf.device('/:cpu:0'):
                    sparsity = get_sparsity_unknown_size(tf.nn.relu(quantized_inputs))
                    store_op = weights["stored_tensor"].assign(quantized_inputs[0])  # TODO REMOVE, DEBUG ONLY
                    sparsity_ops = [weights["sparsity"].assign(sparsity), store_op]

            else:
                sparsity_ops = []

        with tf.control_dependencies(sparsity_ops):
            return tf.identity(quantized_inputs)


# class BFPInputQuantizer(BFPQuantizer):
#     def __init__(self, num_bits):
#         super().__init__(num_bits)
#
#     def __call__(self, inputs, training, weights, **kwargs):
#         quantized_inputs = quantize_variable(variable=inputs,
#                                              exp=weights["exp"],
#                                              width=None,
#                                              clip=False,
#                                              clip_negative=False,
#                                              round_or_floor="floor",
#                                              name="input_quantize")
#
#         return quantized_inputs


class BFPBiasQuantizer(BFPQuantizer):
    def __init__(self, num_bits):
        super().__init__(num_bits)

    def get_config(self):
        config = super().get_config()
        return config

    def build(self, tensor_shape, name, layer):
        variable_dict = super().build(tensor_shape, name, layer)

        variable_dict["stored_tensor"] = layer.add_weight(
            name=name + "_stored_tensor",
            shape=tensor_shape,
            initializer=keras.initializers.zeros(),
            trainable=False,
            dtype=float_policy
        )

        return variable_dict

    def __call__(self, inputs, training, weights, **kwargs):

        # Weights mode and we aren't in training
        # don't need to be re quantized outside training
        if training is False:
            return weights["stored_tensor"]
        else:

            quantized_inputs = quantize_variable(variable=inputs,
                                                 exp=weights["exp"],
                                                 width=self.num_bits,
                                                 clip=True,
                                                 round_or_floor="round",
                                                 name="bias")

            sparsity = get_sparsity_fixed_size(quantized_inputs, self.tensor_size_fp, self.tensor_size_fp_dtype)

            assign_ops = [weights["stored_tensor"].assign(quantized_inputs), weights["sparsity"].assign(sparsity)]

            with tf.control_dependencies(assign_ops):
                quantized_inputs = tf.identity(quantized_inputs)

            return quantized_inputs


class BFPQuantizeConfig(QuantizeConfig):
    def __init__(self, output_quantizer=None, weight_quantizer=None, bias_quantizer=None, pruning_policy=0):

        self.pruning_policy = pruning_policy

        if pruning_policy is not None and (isinstance(pruning_policy, dict) or pruning_policy > 0.0):
            if isinstance(pruning_policy, dict) is False:

                pruning_config = {
                    "initial_sparsity": 0.0,
                    "final_sparsity": pruning_policy,
                    "begin_step": 10,
                    "end_step": 10000,
                    "frequency": 100
                }
                pruning_type = PolynomialDecay
            else:
                pruning_config = pruning_policy["config"]
                pruning_type = pruning_policy["type"]

            enable_pruning = True
            pruning_schedule = pruning_type(**pruning_config)
        else:
            enable_pruning = False
            pruning_schedule = None

        if weight_quantizer is None:
            self.weight_quantizer = BFPWeightQuantizer(num_bits=_WEIGHTS_NUM_BITS,
                                                       enable_pruning=enable_pruning,
                                                       pruning_schedule=pruning_schedule)
        else:
            self.weight_quantizer = weight_quantizer

        if bias_quantizer is None:
            self.bias_quantizer = BFPBiasQuantizer(num_bits=_WEIGHTS_NUM_BITS)
        else:
            self.bias_quantizer = bias_quantizer

        if output_quantizer is None:
            # TODO num batch is hardcoded right now
            self.output_quantizer = BFPActivQuantizer(num_bits=_ACTIV_NUM_BITS, num_batch=int(2 ** 12))
        else:
            self.output_quantizer = output_quantizer

    # Configure how to quantize weights and biases
    def get_weights_and_quantizers(self, layer):


        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            kernel_quant = (layer.depthwise_kernel, self.weight_quantizer)
        else:
            kernel_quant = (layer.kernel, self.weight_quantizer)

        if layer.use_bias:
            return [kernel_quant,
                    (layer.bias, self.bias_quantizer)
                    ]
        else:
            return [kernel_quant]

    def set_quantize_weights(self, layer, quantize_weights):
        # Add this line for each item returned in get_weights_and_quantizers in the same order
        if isinstance(layer, tf.keras.layers.DepthwiseConv2D):
            layer.depthwise_kernel = quantize_weights[0]
        else:
            layer.kernel = quantize_weights[0]

        if layer.use_bias:
            layer.bias = quantize_weights[1]

    # Configure how to quantize activations.
    def get_activations_and_quantizers(self, layer):
        return []

    def set_quantize_activations(self, layer, quantize_activations):
        # Add this line for each item returned in `get_activations_and_quantizers`, in the same order.
        pass

    # Configure how to quantize outputs.
    def get_output_quantizers(self, layer):
        return [self.output_quantizer]

    def get_config(self):
        config = dict()
        config["output_quantizer"] = self.output_quantizer
        config["weight_quantizer"] = self.weight_quantizer
        config["bias_quantizer"] = self.bias_quantizer
        config["pruning_policy"] = self.pruning_policy
        return config


# class BFPInputQuantizerConfig(QuantizeConfig):
#     def __init__(self, output_quantizer=None, ):
#         if output_quantizer is None:
#             self.output_quantizer = BFPInputQuantizer(num_bits=8)
#         else:
#             self.output_quantizer = output_quantizer
#
#     # Configure how to quantize weights and biases
#     def get_weights_and_quantizers(self, layer):
#         return []
#
#     def set_quantize_weights(self, layer, quantize_weights):
#         # Add this line for each item returned in get_weights_and_quantizers in the same order
#         pass
#
#     # Configure how to quantize activations.
#     def get_activations_and_quantizers(self, layer):
#         return []
#
#     def set_quantize_activations(self, layer, quantize_activations):
#         # Add this line for each item returned in `get_activations_and_quantizers`, in the same order.
#         pass
#
#     # Configure how to quantize outputs.
#     def get_output_quantizers(self, layer):
#         return [self.output_quantizer]
#
#     def get_config(self):
#         config = dict()
#         config["output_quantizer"] = self.output_quantizer
#         return config
#

class QuantizerSaveCallback(Callback):

    def __init__(self, filepath):
        self.filepath = filepath + r"/quantized//"
        os.mkdir(self.filepath)

    def on_epoch_end(self, epoch, logs=None):

        savedir = self.filepath + "/epoch_" + str(epoch) + "/"
        os.mkdir(savedir)

        shift_per_layer_kernel = list()
        # TODO . Name should be changed to output
        shift_per_layer_activation = [8]  # first layer in TNH has always shift of 8

        shift_per_layer_bias = list()

        for layer in self.model.layers:
            if isinstance(layer, QuantizeWrapper):
                layer_name = layer.name
                all_variables = layer.weights
                to_save = {}

                kernel_exp_found = False
                output_exp_found = False
                bias_exp_found = False

                for variable in all_variables:
                    if "kernel_stored_tensor" in variable.name:
                        to_save["kernel"] = variable
                    elif "bias_stored_tensor" in variable.name:
                        to_save["bias"] = variable

                    elif "kernel_exp" in variable.name:
                        if kernel_exp_found is False:
                            kernel_exp_found = True
                            shift_per_layer_kernel.append(variable.numpy())
                    elif "bias_exp" in variable.name:
                        if bias_exp_found is False:
                            bias_exp_found = True
                            shift_per_layer_bias.append(variable.numpy())
                    elif "output_exp" in variable.name:
                        if output_exp_found is False:
                            output_exp_found = True
                            shift_per_layer_activation.append(variable.numpy())

                assert len(to_save) == 2
                assert len(shift_per_layer_activation) - 1 == len(shift_per_layer_bias) == len(shift_per_layer_kernel)

                full_path = savedir + "shift_per_layer_activation"
                np.save(file=full_path, arr=shift_per_layer_activation)

                full_path = savedir + "shift_per_layer_bias"
                np.save(file=full_path, arr=shift_per_layer_bias)

                full_path = savedir + "shift_per_layer_kernel"
                np.save(file=full_path, arr=shift_per_layer_kernel)

                full_name = (layer_name + "_shifts").replace("quant_", "")
                full_path = savedir + full_name
                np.save(file=full_path, arr=[shift_per_layer_kernel[-1], shift_per_layer_bias[-1],
                                             shift_per_layer_activation[-1]])

                full_name = (layer_name + "_kernel").replace("quant_", "")
                full_path = savedir + full_name
                value = (to_save["kernel"].numpy() * (2.0 ** float(shift_per_layer_kernel[-1]))).astype(np.int32)
                np.save(file=full_path, arr=value)

                full_name = (layer_name + "_bias").replace("quant_", "")
                full_path = savedir + full_name
                value = (to_save["bias"].numpy() * (2.0 ** float(shift_per_layer_bias[-1]))).astype(np.int32)
                np.save(file=full_path, arr=value)



def apply_quantization(model, pruning_policy=None, weight_precision=None, activation_precision=None):
    # assert quantize_base.SET_CUSTOM_TNH_FLAG, log.info("TFMOD needs to be modified with quantizer disabled for proper "
    #                                                    "running")

    if weight_precision is not None:
        global _WEIGHTS_NUM_BITS  # need to declare when you want to change the value
        _WEIGHTS_NUM_BITS = weight_precision

    if activation_precision is not None:
        global _ACTIV_NUM_BITS
        _ACTIV_NUM_BITS = activation_precision


    # Helper function uses `quantize_annotate_layer` to annotate that only the
    # Dense layers should be quantized.
    def add_quantize_annotation(layer):
        # create new layer to break link with old model
        try:
            layer = layer.__class__.from_config(layer.get_config())
        except:
            pass



        for layer_type in quantization_map:

            if isinstance(layer, layer_type):

                if isinstance(pruning_policy, float) or pruning_policy is None:
                    layer_pruning = pruning_policy
                elif isinstance(pruning_policy, dict):
                    layer_pruning = pruning_policy[layer.name]
                else:
                    raise ValueError("Illegal layer pruning policy {}".format(pruning_policy))

                quantize_config = BFPQuantizeConfig(pruning_policy=layer_pruning)

                log.info(
                    "**Quantization annotation added to layer {} of type {} with {}".format(layer.name,
                                                                                            layer_type,
                                                                                            quantize_config))

                quantized_layer = quantize_annotate_layer(to_annotate=layer, quantize_config=quantize_config)
                return quantized_layer
        log.info("**Quantization annotation not added to layer {} of type {}".format(layer.name, type(layer)))

        return layer

    # Use `tf.keras.models.clone_model` to apply `add_quantize_annotation`
    # to the layers of the model.
    log.info("Annotating model {}".format(model.name))

    tf.keras.backend.clear_session()
    annotated_model = tf.keras.models.clone_model(
        model,
        clone_function=add_quantize_annotation,
    )



    with quantize_scope({
        'BFPQuantizeConfig': BFPQuantizeConfig,
        "BFPActivQuantizer": BFPActivQuantizer,
        "BFPWeightQuantizer": BFPWeightQuantizer,
        "BFPBiasQuantizer": BFPBiasQuantizer,
        "PolynomialDecay": PolynomialDecay
    }):
        # Use `quantize_apply` to actually make the model quantization aware.
        quant_aware_model = quantize_apply(annotated_model)

    for q_layer in quant_aware_model.layers:
        if isinstance(q_layer, QuantizeWrapper):
            for quant_type in quantization_map:
                if isinstance(q_layer.layer, quant_type):
                    original_name = q_layer.name.replace("quant_", "")
                    old_layer = model.get_layer(original_name)

                    q_weights = q_layer.get_weights()
                    orig_weights = old_layer.get_weights()

                    q_weights[0] = orig_weights[0]
                    try:
                        q_weights[1] = orig_weights[1]
                    except IndexError:
                        pass
                    q_layer.set_weights(q_weights)


    return quant_aware_model
