import numpy as np
from utils import *


class DataSet:
    """"""

    def __init__(self, batch_size, train_amount, test_amount, valid_amount, reshape=True, seed=8):
        """"""
        global_seed(seed)
        train, test, valid, self.stats = get_split_dataset(train_amount,
                                                           test_amount,
                                                           valid_amount)
        train_examples = [list(column) for column in zip(*[d.values() for d in train])]
        test_examples  = [list(column) for column in zip(*[d.values() for d in test])]
        valid_examples = [list(column) for column in zip(*[d.values() for d in valid])]
        self.train_size = len(train)
        self.test_size = len(test)
        self.valid_size = len(valid)
        self.train_x = np.array(train_examples[1])
        self.train_y = np.array(train_examples[0])
        self.train_len = np.array(train_examples[3])
        self.test_x = np.array(test_examples[1])
        self.test_y = np.array(test_examples[0])
        self.test_len = np.array(test_examples[3])
        self.valid_x = np.array(valid_examples[1])
        self.valid_y = np.array(valid_examples[0])
        self.valid_len = np.array(valid_examples[3])
        if (reshape):
            self.train_x = self.train_x.reshape((-1, self.stats['max_sequence_length'], 1))
            self.valid_x = self.valid_x.reshape((-1, self.stats['max_sequence_length'], 1))
            self.test_x  = self.test_x.reshape((-1, self.stats['max_sequence_length'], 1))
        self.train_batch_counter = 0
        self.test_batch_counter = 0
        self.valid_batch_counter = 0
        self.batch_size = batch_size

    def get_valid_set(self):
        """"""
        return self.valid_x, self.valid_y, self.valid_len

    def get_train_set(self):
        """"""
        return self.train_x, self.train_y, self.train_len

    def get_test_set(self):
        """"""
        return self.test_x, self.test_y, self.test_len

    def get_number_train_batches(self):
        """"""
        return int(np.floor(float(self.train_size / self.batch_size)))

    def get_number_valid_batches(self):
        """"""
        return int(np.floor(float(self.valid_size / self.batch_size)))

    def get_number_test_batches(self):
        """"""
        return int(np.floor(float(self.test_size / self.batch_size)))

    def get_next_train_batch(self):
        """"""
        start = self.train_batch_counter * self.batch_size
        end = (self.train_batch_counter + 1) * self.batch_size
        self.train_batch_counter = (self.train_batch_counter + 1) % (self.get_number_train_batches() - 1)
        return self.train_x[start:end], self.train_y[start:end], self.train_len[start:end]

    def get_next_test_batch(self):
        """"""
        start = self.test_batch_counter * self.batch_size
        end = (self.test_batch_counter + 1) * self.batch_size
        self.test_batch_counter = (self.test_batch_counter + 1) % (self.get_number_test_batches() - 1)
        return self.test_x[start:end], self.test_y[start:end], self.test_len[start:end]

    def get_next_valid_batch(self):
        """"""
        start = self.valid_batch_counter * self.batch_size
        end = (self.valid_batch_counter + 1) * self.batch_size
        self.valid_batch_counter = (self.valid_batch_counter + 1) % (self.get_number_valid_batches() - 1)
        return self.valid_x[start:end], self.valid_y[start:end], self.valid_len[start:end]
