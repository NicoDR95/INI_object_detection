import tensorflow as tf


class Optimizer(object):
    def __init__(self, parameters):
        self.parameters = parameters

    def get_optimizer(self):
        self.optimizer_object = tf.train.AdamOptimizer(learning_rate=self.parameters.learning_rate)
        return self.optimizer_object
