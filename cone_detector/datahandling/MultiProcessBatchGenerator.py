import logging
from multiprocessing import Process, Pipe

from datahandling.BatchGenerator import BatchGenerator

log = logging.getLogger()


def async_batch_reader(generator_class, pipe_send_end):
    while True:
        # Initialize the generator
        generator = generator_class.get_generator()
        # Iterate over it putting it in a queue

        for batch in generator:
            # The put method is a blocking one. If the queue is full, it stop.
            # Since we dont set a timeout it will block forever (that should not happen since we keep training) without losing data
            pipe_send_end.send(batch)

        pipe_send_end.send(None)


class MultiProcessBatchGenerator(BatchGenerator):
    def __init__(self, parameters, dataset=None, preprocessor=None, visualizer=None, store_batch_y=True):
        super(MultiProcessBatchGenerator, self).__init__(parameters, dataset, preprocessor, visualizer, store_batch_y)
        self.async_reader_process = None
        self.started = False
        self.batch_pipe_sender, self.batch_pipe_receiver = Pipe()

    def get_generator(self):

        if self.started is False:
            self.async_reader_process = Process(target=async_batch_reader, args=(super(), self.batch_pipe_sender))
            self.async_reader_process.start()
            self.started = True

        while True:
            next_batch = self.batch_pipe_receiver.recv()

            if next_batch is None:
                break

            yield next_batch
