import tensorflow as tf
import numpy as np


class Pruning(object):

    def __init__(self, parameters, tensor_shape, enable_pruning, pruning_config, layer_name):
        self.parameters = parameters
        self.enable_pruning = enable_pruning
        self.pruning_schedule = PolynomialDecay(initial_sparsity=pruning_config["initial_sparsity"],
                                                final_sparsity=pruning_config["final_sparsity"],
                                                begin_step=pruning_config["begin_step"],
                                                end_step=pruning_config["end_step"],
                                                frequency=pruning_config["frequency"])
        self.tensor_size_int = None
        self.tensor_size_fp = None
        self.tensor_size_fp_dtype = None
        self.mask_name = layer_name + '_mask'
        self.step_name = layer_name + '_pruning_step'
        # Initialize weights with only the pruning step to 0
        with tf.device('/cpu:0'):
            self.pruning_step = tf.Variable(initial_value=0, trainable=False, dtype=tf.int32, name=self.step_name)
        # print('pruning step:', self.pruning_step)
        # Initialize mask to all ones
        initial_mask = np.ones(shape=tensor_shape, dtype=np.float32)
        self.mask = tf.Variable(initial_value=initial_mask, trainable=False, dtype=tf.float32, name=self.mask_name)
        # print('initial mask:', self.mask)

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

        ### NEW! ###
        self.pruning_tensor_size_int = 1
        for dim in tensor_shape[0:-1]:
            self.pruning_tensor_size_int = self.pruning_tensor_size_int * dim

        if self.pruning_tensor_size_int <= 2 ** 10:
            self.pruning_tensor_size_fp_dtype = tf.float16
        else:
            self.pruning_tensor_size_fp_dtype = tf.float32

        self.pruning_tensor_size_fp = tf.dtypes.cast(self.pruning_tensor_size_int,
                                                     dtype=self.pruning_tensor_size_fp_dtype)

    ### NEW FUNCTION! ###
    def get_pruning_threshold_index(self, sparsity):
        # Compute position of threshold in the array.r when sparsity is  Must be capped to avoid erro0
        density = tf.cast((1 - sparsity), self.pruning_tensor_size_fp_dtype)
        pruning_threshold_pre_round = self.pruning_tensor_size_fp * density

        pruning_threshold_uncapped = tf.math.ceil(pruning_threshold_pre_round)
        pruning_threshold_index_uncapped = tf.dtypes.cast(pruning_threshold_uncapped, tf.int32)

        pruning_threshold_index = tf.math.minimum(pruning_threshold_index_uncapped, self.pruning_tensor_size_int - 1)

        return pruning_threshold_index

    ### NEW FUNCTION! ###
    def update_mask_kernel(self, abs_kernel, pruning_threshold_index):
        # Sort the entire array (flattened)
        sorted_weights, _ = tf.math.top_k(tf.reshape(abs_kernel, [-1]), k=self.pruning_tensor_size_int)

        # Selected the threshold value
        current_threshold = sorted_weights[pruning_threshold_index]

        # compare to get new mask
        mask_bool = tf.math.greater_equal(abs_kernel, current_threshold)
        mask = tf.dtypes.cast(mask_bool, abs_kernel.dtype)

        return mask

    def get_pruned_kernel_old(self, inputs, training):

        if self.enable_pruning is True:
            # Get absolute values array
            abs_weights = tf.abs(inputs)

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

                # assign_ops.append(self.weights["mask"].assign(mask))
                # assign_ops.append(self.weights["threshold"].assign(current_threshold))

                return mask

            # Get new sparsity level
            update_pruning, sparsity = self.pruning_schedule(self.pruning_step)
            # print(self.mask)
            sel_mask = tf.cond(update_pruning, lambda: update_mask(sparsity), lambda: self.mask)

            mask_update = self.mask.assign(sel_mask)
            no_grad_mask = tf.stop_gradient(sel_mask)
            with tf.device('/cpu:0'):
                pruning_update = self.pruning_step.assign(self.pruning_step + 1)
            with tf.control_dependencies([mask_update, pruning_update]):
                masked_weight = tf.math.multiply(inputs, no_grad_mask)
            # updated even if not pruning since we need it for keeping track of when to prune
            # assign_ops.append(self.weights["pruning_step"].assign(incremented_pruning_step))

        else:
            masked_weight = inputs

        return masked_weight

    ### NEW FUNCTION! ###
    def get_pruned_kernel(self, inputs, training):
        # tf.print(self.name + "\n")
        # Weights mode and we aren't in training
        # don't need to be re quantized outside training
        # assign_ops = []

        if self.enable_pruning is True:
            with tf.name_scope("pruning"):
                # Get quantization exp and absoluted array
                abs_weights = tf.abs(inputs)

                # Get new sparsity level
                update_pruning, sparsity = self.pruning_schedule(self.pruning_step)

                def update_mask():
                    # TODO FIX FOR FC
                    reordered_kernel = tf.transpose(abs_weights, [3, 1, 2, 0])
                    pruning_threshold_index = self.get_pruning_threshold_index(sparsity)


                    reordered_mask = tf.map_fn(lambda x: self.update_mask_kernel(x, pruning_threshold_index), reordered_kernel,
                                               parallel_iterations=int(inputs.shape[-1]), back_prop=False, swap_memory=True,
                                               infer_shape=True, name=None)

                    mask = tf.transpose(reordered_mask, [3, 1, 2, 0])

                    # assign_ops.append(weights["mask"].assign(mask))

                    return mask

                sel_mask = tf.cond(update_pruning, lambda: update_mask(), lambda: self.mask)
                no_grad_mask = tf.stop_gradient(sel_mask)

                mask_update = self.mask.assign(sel_mask)
                with tf.device('/cpu:0'):
                    pruning_update = self.pruning_step.assign(self.pruning_step + 1)
                with tf.control_dependencies([mask_update, pruning_update]):
                    masked_weight = tf.math.multiply(inputs, no_grad_mask)

                # updated even if not pruning since we need it for keeping track of when to prune
                # assign_ops.append(weights["pruning_step"].assign(incremented_pruning_step))

        else:
            # Get quantization exp and absoluted array
            masked_weight = inputs

            # sparsity = get_sparsity_fixed_size(quantized_inputs, self.tensor_size_fp, self.tensor_size_fp_dtype)
            #
            # assign_ops.append(weights["sparsity"].assign(sparsity))
            # assign_ops.append(weights["stored_tensor"].assign(quantized_inputs))
            # assign_ops.append(weights["exp"].assign(quant_exp))

        return masked_weight



# ██████╗ ██████╗ ██╗   ██╗███╗   ██╗██╗███╗   ██╗ ██████╗     ██╗     ██╗██████╗     ███████╗██╗   ██╗███╗   ██╗ ██████╗████████╗██╗ ██████╗ ███╗   ██╗███████╗
# ██╔══██╗██╔══██╗██║   ██║████╗  ██║██║████╗  ██║██╔════╝     ██║     ██║██╔══██╗    ██╔════╝██║   ██║████╗  ██║██╔════╝╚══██╔══╝██║██╔═══██╗████╗  ██║██╔════╝
# ██████╔╝██████╔╝██║   ██║██╔██╗ ██║██║██╔██╗ ██║██║  ███╗    ██║     ██║██████╔╝    █████╗  ██║   ██║██╔██╗ ██║██║        ██║   ██║██║   ██║██╔██╗ ██║███████╗
# ██╔═══╝ ██╔══██╗██║   ██║██║╚██╗██║██║██║╚██╗██║██║   ██║    ██║     ██║██╔══██╗    ██╔══╝  ██║   ██║██║╚██╗██║██║        ██║   ██║██║   ██║██║╚██╗██║╚════██║
# ██║     ██║  ██║╚██████╔╝██║ ╚████║██║██║ ╚████║╚██████╔╝    ███████╗██║██████╔╝    ██║     ╚██████╔╝██║ ╚████║╚██████╗   ██║   ██║╚██████╔╝██║ ╚████║███████║
# ╚═╝     ╚═╝  ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚═╝╚═╝  ╚═══╝ ╚═════╝     ╚══════╝╚═╝╚═════╝     ╚═╝      ╚═════╝ ╚═╝  ╚═══╝ ╚═════╝   ╚═╝   ╚═╝ ╚═════╝ ╚═╝  ╚═══╝╚══════╝

"""Pruning Schedule classes to control pruning rate during training."""

import abc
import six
import tensorflow as tf


@six.add_metaclass(abc.ABCMeta)
class PruningSchedule(object):
    """Specifies when to prune layer and the sparsity(%) at each training step.

  PruningSchedule controls pruning during training by notifying at each step
  whether the layer's weights should be pruned or not, and the sparsity(%) at
  which they should be pruned.

  It can be invoked as a `callable` by providing the training `step` Tensor. It
  returns a tuple of bool and float tensors.

  ```python
    should_prune, sparsity = pruning_schedule(step)
  ```

  You can inherit this class to write your own custom pruning schedule.
  """

    # @staticmethod
    # def floormod(a, b):
    #     floormod = tf.math.subtract(a, tf.math.multiply(b, tf.math.floordiv(a, b)))
    #
    #     return floormod

    @staticmethod
    def _should_prune_in_step(step, begin_step, end_step, frequency):
        """Checks if pruning should be applied in the current training step.

    Pruning should only occur within the [`begin_step`, `end_step`] range every
    `frequency` number of steps.

    Args:
      step: Current training step.
      begin_step: Step at which to begin pruning.
      end_step: Step at which to end pruning.
      frequency: Only apply pruning every `frequency` steps.

    Returns:
      True/False, if pruning should be applied in current step.
    """
        is_in_pruning_range = tf.math.logical_and(
            tf.math.greater_equal(step, begin_step),
            # If end_pruning_step is negative, keep pruning forever!
            tf.math.logical_or(
                tf.math.less_equal(step, end_step), tf.math.less(end_step, 0)))

        is_pruning_turn = tf.math.equal(tf.floormod(tf.math.subtract(step, begin_step), frequency), 0)

        # a = tf.math.subtract(step, begin_step)
        # b = frequency
        # floormod = tf.math.subtract(a, tf.math.multiply(b, tf.math.floordiv(a, b)))
        # is_pruning_turn = tf.math.equal(floormod, 0)

        return tf.math.logical_and(is_in_pruning_range, is_pruning_turn)

    @staticmethod
    def _validate_step(begin_step, end_step, frequency, allow_negative_1):
        """Checks whether the parameters for pruning schedule are valid.

    Args:
      begin_step: Step at which to begin pruning.
      end_step: Step at which to end pruning. Special value of `-1` implies
        pruning can continue forever.
      frequency: Only apply pruning every `frequency` steps.
      allow_negative_1: Whether end_step is allowed to be `-1` or not.

    Returns:
      None
    """

        if begin_step < 0:
            raise ValueError('begin_step should be >= 0')

        # In cases like PolynomialDecay, continuing to prune forever does not make
        # sense. The function needs an end_step to decay the sparsity.
        if not allow_negative_1 and end_step == -1:
            raise ValueError('end_step cannot be -1.')

        if end_step != -1:
            if end_step < 0:
                raise ValueError('end_step can be -1 or >= 0')
            if end_step < begin_step:
                raise ValueError('begin_step should be <= end_step if end_step != -1')

        if frequency <= 0:
            raise ValueError('frequency should be > 0')

    @staticmethod
    def _validate_sparsity(sparsity, variable_name):
        if not 0.0 <= sparsity < 1.0:
            raise ValueError('{} must be in range [0,1)'.format(variable_name))

    @abc.abstractmethod
    def __call__(self, step):
        """Returns the sparsity(%) to be applied.

    If the returned sparsity(%) is 0, pruning is ignored for the step.

    Args:
      step: Current step in graph execution.

    Returns:
      Sparsity (%) that should be applied to the weights for the step.
    """
        raise NotImplementedError(
            'PruningSchedule implementation must override __call__')

    @abc.abstractmethod
    def get_config(self):
        raise NotImplementedError(
            'PruningSchedule implementation override get_config')

    @classmethod
    def from_config(cls, config):
        """Instantiates a `PruningSchedule` from its config.

    Args:
        config: Output of `get_config()`.

    Returns:
        A `PruningSchedule` instance.
    """
        return cls(**config)


class PolynomialDecay(PruningSchedule):
    """Pruning Schedule with a PolynomialDecay function."""

    def __init__(self,
                 initial_sparsity,
                 final_sparsity,
                 begin_step,
                 end_step,
                 power=3,
                 frequency=100):
        """Initializes a Pruning schedule with a PolynomialDecay function.

    Pruning rate grows rapidly in the beginning from initial_sparsity, but then
    plateaus slowly to the target sparsity. The function applied is

    current_sparsity = final_sparsity + (initial_sparsity - final_sparsity)
          * (1 - (step - begin_step)/(end_step - begin_step)) ^ exponent

    which is a polynomial decay function. See
    [paper](https://arxiv.org/abs/1710.01878).

    Args:
      initial_sparsity: Sparsity (%) at which pruning begins.
      final_sparsity: Sparsity (%) at which pruning ends.
      begin_step: Step at which to begin pruning.
      end_step: Step at which to end pruning.
      power: Exponent to be used in the sparsity function.
      frequency: Only apply pruning every `frequency` steps.
    """

        self.initial_sparsity = initial_sparsity
        self.final_sparsity = final_sparsity
        self.power = power

        self.begin_step = begin_step
        self.end_step = end_step
        self.frequency = frequency

        self._validate_step(self.begin_step, self.end_step, self.frequency, False)
        self._validate_sparsity(initial_sparsity, 'initial_sparsity')
        self._validate_sparsity(final_sparsity, 'final_sparsity')

    def __call__(self, step):
        # TODO(tf-mot): consider switch to divide for 1.XX also.
        if hasattr(tf, 'div'):
            divide = tf.div
        else:
            divide = tf.math.divide

        # TODO(pulkitb): Replace function with tf.polynomial_decay
        with tf.name_scope('polynomial_decay_pruning_schedule'):
            p = tf.math.minimum(
                1.0,
                tf.math.maximum(
                    0.0,
                    divide(
                        tf.dtypes.cast(step - self.begin_step, tf.float32),
                        self.end_step - self.begin_step)))
            sparsity = tf.math.add(
                tf.math.multiply(self.initial_sparsity - self.final_sparsity,
                                 tf.math.pow(1 - p, self.power)),
                self.final_sparsity,
                name='sparsity')

        return (self._should_prune_in_step(step, self.begin_step, self.end_step,
                                           self.frequency),
                sparsity)

    def get_config(self):
        return {
            'class_name': self.__class__.__name__,
            'config': {
                'initial_sparsity': self.initial_sparsity,
                'final_sparsity': self.final_sparsity,
                'power': self.power,
                'begin_step': self.begin_step,
                'end_step': self.end_step,
                'frequency': self.frequency
            }
        }
