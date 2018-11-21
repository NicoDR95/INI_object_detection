import logging
import math
import time
import traceback

import os
import tensorflow as tf

from utility.utility_library import measure_time

log = logging.getLogger()


class TrainPipeline(object):
    def __init__(self, parameters, dataset, data_preprocessing, batch_generator,
                 network, loss, optimizer_object, visualizer, accuracy):
        self.parameters = parameters
        self.dataset = dataset
        self.data_preprocessing = data_preprocessing
        self.batch_generator = batch_generator
        self.network = network
        self.loss = loss
        self.optimizer_object = optimizer_object
        self.visualizer = visualizer
        self.accuracy = accuracy
        self.training_prepared = False
        self.network_loss_linked = False
        self.training_data_prepared = False
        self.tf_runned = False

    def train(self):
        if self.training_data_prepared is False:
            self.all_datasets_dict, self.global_step, self.epoch_step = self.prepare_training()

        if self.network_loss_linked is False:
            self.link_network_loss()

        if self.tf_runned is False:
            self.merged_summary_op, self.summary_writer, self.sess, self.saver = self.get_tf_run()

        increment_op = tf.assign_add(self.epoch_step, 1, name='increment_epoch_step')

        op_to_run = [self.optimizer, self.merged_summary_op, self.loss_tf]

        self.batch_generator.set_dataset(dataset=self.all_datasets_dict,
                                         preprocessor=self.data_preprocessing,
                                         visualizer=self.visualizer)

        for epoch in range(self.parameters.n_epochs):
            try:
                if 'mid_epoch' not in tf.train.latest_checkpoint(self.parameters.saved_model_dir):
                    epoch_n = self.sess.run(increment_op)
                else:
                    epoch_n = self.sess.run(self.epoch_step)
            except TypeError:
                epoch_n = self.sess.run(increment_op)

            batches = self.batch_generator.get_generator()

            try:
                epoch_start_t = time.time()
                for batch_iter_counter, (batch_images, batch_true_val, filenames) in enumerate(batches):
                    batch_start_t = time.time()

                    _, summary, loss = self.sess.run(op_to_run, feed_dict={self.input_ph: batch_images,
                                                                           self.true_values_ph: batch_true_val,
                                                                           self.train_flag_ph: self.parameters.training})

                    if loss > self.parameters.loss_filename_print_threshold or math.isnan(loss):
                        log.warn("Following images gave loss higher than conf_threshold: {}".format(filenames))

                    self.summary_writer.add_summary(summary, self.sess.run(self.global_step))
                    self.summary_writer.flush()

                    batch_end_t = time.time()
                    batch_time = batch_end_t - batch_start_t
                    img_s = self.parameters.batch_size / batch_time
                    percent_batches_done = 100 * batch_iter_counter / self.batch_generator.num_batches
                    log.info("Epoch {} - Batch {}/{} ({:.2f}%) - time: {:.2f} ({:.2f} img/s) - loss {:.5f}".format(epoch_n, batch_iter_counter + 1,
                                                                                                                   self.batch_generator.num_batches,
                                                                                                                   percent_batches_done,
                                                                                                                   batch_time, img_s, loss))


            except KeyboardInterrupt:
                log.info("Keyboard interrupt received")
                log.info("Saving the models in 3 seconds...")
                log.info("Press ctrl+C again to abort")
                time.sleep(3)

                self.run_accuracy(epoch_n, False)

                log.info("Saving the model...")

                if not os.path.exists(self.parameters.saved_model_dir):
                    log.info("Creating save model dir {}".format(self.parameters.saved_model_dir))
                    os.makedirs(self.parameters.saved_model_dir)

                # self.summary_writer.flush()
                with tf.control_dependencies(self.summary_writer.flush()):
                    self.saver.save(self.sess, os.path.join(self.parameters.saved_model_dir, self.parameters.saved_model_name + '-mid_epoch'),
                                    global_step=epoch_n)

                exit("Model saved")
            epoch_end_t = time.time()
            log.info("######################### Epoch {} completed #############################".format(epoch_n))
            log.info("Epoch time: {:.2f}".format(epoch_end_t - epoch_start_t))
            log.info("Saving the model...")

            if not os.path.exists(self.parameters.saved_model_dir):
                log.info("Creating save model dir {}".format(self.parameters.saved_model_dir))
                os.makedirs(self.parameters.saved_model_dir)
            # self.summary_writer.flush()
            with tf.control_dependencies(self.summary_writer.flush()):
                self.saver.save(self.sess, os.path.join(self.parameters.saved_model_dir, self.parameters.saved_model_name),
                                global_step=epoch_n)

            self.run_accuracy(epoch_n, True)

        log.info("Requested epochs done - training completed")

    @measure_time
    def run_accuracy(self, step, step_finished):
        try:
            log.info("Evaluating results...")
            self.accuracy.run_and_get_accuracy(train_sess=self.sess, step=step, epoch_finished=step_finished)
        except Exception as e:
            log.error("**********************Exception during get accuracy******************")
            log.error("Error:" + str(e))
            traceback.print_exc()

    def prepare_training(self):
        with tf.device("/cpu:0"):
            with tf.name_scope(name='training_steps'):
                global_step = tf.Variable(0, name='global_step', trainable=False)
                epoch_step = tf.Variable(0, name='epoch_step', trainable=False)

        # self.dataset is a list of datasets, we'll iterate on them and put them is single list of dictionaries
        all_dataset_dict = list()
        for reader in self.dataset:
            new_annotation = reader.get_dataset_dict()
            if new_annotation is not None:
                all_dataset_dict.extend(new_annotation)

        # TODO, pass the augmented data as a second dataset calculated offline
        # augmented_data = self.data_augmentation.get_augemnted_data(all_dataset_dict)
        # TODO, prreprocess the data one batch at time in the batch generator
        # preprocessed_data = self.data_preprocessing.get_preprocessed_data(all_dataset_dict)
        self.training_data_prepared = True

        return all_dataset_dict, global_step, epoch_step

    def link_network_loss(self):

        self.net_output, self.input_ph, self.train_flag_ph = self.network.get_network()
        self.loss_tf, self.true_values_ph = self.loss.get_loss(self.net_output)
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)

        with tf.control_dependencies(update_ops):
            self.optimizer = self.optimizer_object.get_optimizer().minimize(self.loss_tf, global_step=self.global_step)

        self.network_loss_linked = True
        log.info('link_network_loss completed')

    def get_epoch_step(self):
        step = tf.train.latest_checkpoint(self.parameters.saved_model_dir)
        if step is not None:
            step = int(step[-1])
            # Old and not flexible:
            # step_index = step.index(self.parameters.saved_model_name + '-')
            # step_index = step_index + len(self.parameters.saved_model_name + '-')
            # step = step[step_index:]
            # step = int(step)
        else:
            step = 0

        return step

    def get_tf_run(self):

        # epoch_step = self.get_epoch_step()

        sess = tf.Session()
        sess.run(tf.global_variables_initializer())

        # The saver must be on the cpu
        with tf.device("/cpu:0"):
            saver = tf.train.Saver(max_to_keep=self.parameters.checkpoints_to_keep)

            if tf.train.latest_checkpoint(self.parameters.saved_model_dir) is not None:
                saver.restore(sess, tf.train.latest_checkpoint(self.parameters.saved_model_dir))

            # Tensorboard Stuff
            tf.summary.scalar("loss", self.loss_tf)
            merged_summary_op = tf.summary.merge_all()
            summary_writer = tf.summary.FileWriter(self.parameters.tensorboard_dir, sess.graph, flush_secs=30)

        self.tf_runned = True

        return merged_summary_op, summary_writer, sess, saver
