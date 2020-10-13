import tensorflow as tf
from networks.NetworkBase import NetworkBase


class NetworkBaseSparsityAnn (NetworkBase):

    def get_network(self):
        self.set_placeholders()
        net_output, sparsity_dict = self.network_build(self.input_ph)

        return net_output, self.input_ph, self.train_flag_ph, sparsity_dict
