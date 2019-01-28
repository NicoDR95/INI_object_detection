
import logging
from multiprocessing import Process, Pipe

from datahandling.BatchGenerator import BatchGenerator

log = logging.getLogger()


def async_batch_reader(generator_string, pipe_send_end):
    parameters, dataset, preprocessor, visualizer, store_batch_y, dataset,preprocessor, visualizer = generator_string
    generator_class = BatchGenerator(parameters, dataset, preprocessor, visualizer, store_batch_y)
    generator_class.set_dataset(dataset,preprocessor, visualizer)
    while True:
        # Initialize the generator
        generator = generator_class.get_generator()
        # Iterate over it putting it in a queue

        for batch in generator:
            # The put method is a blocking one. If the queue is full, it stop.
            # Since we dont set a timeout it will block forever (that should not happen since we keep training) without losing data
            pipe_send_end.send(batch)

        pipe_send_end.send(None)


#The class extends the base batch generator only for purposes of inerithing the methods and being 1 to 1 compatible
#The actual generator that is used runs in the async thread. Thus any method querying for run time info will fail to
#produce correct results
class MultiProcessBatchGenerator(BatchGenerator):
    def __init__(self, parameters, dataset=None, preprocessor=None, visualizer=None, store_batch_y=True):
        super(MultiProcessBatchGenerator, self).__init__(parameters, dataset, preprocessor, visualizer=None, store_batch_y=False)

        self.batch_gen_init_args = [parameters, dataset, preprocessor, visualizer, store_batch_y]

        self.async_reader_process = None
        self.started = False
        self.batch_pipe_sender, self.batch_pipe_receiver = Pipe()

    def set_dataset(self, dataset=None, preprocessor=None, visualizer=None):
        self.batch_gen_dataset_args = []
        self.batch_gen_dataset_args.append(dataset)
        self.batch_gen_dataset_args.append(preprocessor)
        self.batch_gen_dataset_args.append(None)
        log.warn("MultiProcessBatchGenerator doesnt support visualization")
        super(MultiProcessBatchGenerator, self).set_dataset(dataset, preprocessor ,None)

    def get_generator(self):

        if self.started is False:
            async_arg = self.batch_gen_init_args + self.batch_gen_dataset_args
            self.async_reader_process = Process(target=async_batch_reader, args=(async_arg, self.batch_pipe_sender))
            self.async_reader_process.start()
            self.started = True

        while True:
            next_batch = self.batch_pipe_receiver.recv()

            if next_batch is None:
                break

            yield next_batch
