import tensorflow as tf

class GlobalStep(object):
    def __init__(self, saved_model_path, saved_model_name):
        self.saved_model_path = saved_model_path
        self.saved_model_name = saved_model_name

    def get_global_step(self):
        step = tf.train.latest_checkpoint(self.saved_model_path)
        if step is not None:
            step_index = step.index(self.saved_model_name + '-')
            step_index = step_index + len(self.saved_model_name + '-')
            step = step[step_index:]
            step = int(step)
        else:
            step = 0

        return step
