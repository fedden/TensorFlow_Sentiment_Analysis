import numpy as np
from utils import *

class DataSet:
    """"""

    def __init__(self, batch_size, train_amount, test_amount, valid_amount, seed):
        """"""
        global_seed(seed)
        train, test, valid, self.stats = get_split_dataset(train_amount,
                                                           test_amount,
                                                           valid_amount)
        train_examples = [list(column) for column in zip(*[d.values() for d in train])]
        test_examples  = [list(column) for column in zip(*[d.values() for d in test])]
        valid_examples = [list(column) for column in zip(*[d.values() for d in valid])]
        self.train_size = len(train)
        self.train_x = np.array(train_examples[1]).reshape((-1,
                                                           self.stats['max_sequence_length'],
                                                           1))
        self.train_y = np.array(train_examples[0])
        self.train_len = np.array(train_examples[3])
        self.test_x = np.array(test_examples[1]).reshape((-1,
                                                         self.stats['max_sequence_length'],
                                                         1))
        self.test_y = np.array(test_examples[0])
        self.test_len = np.array(test_examples[3])
        self.valid_x = np.array(valid_examples[1]).reshape((-1,
                                                           self.stats['max_sequence_length'],
                                                           1))
        self.valid_y = np.array(valid_examples[0])
        self.valid_len = np.array(valid_examples[3])
        self.batch_counter = 0
        self.batch_size = batch_size

    def get_number_train_batches(self):
        """"""
        return int(np.floor(float(self.train_size / self.batch_size)))

    def get_valid_set(self):
        """"""
        return self.valid_x, self.valid_y, self.valid_len

    def get_test_set(self):
        """"""
        return self.test_x, self.test_y, self.test_len

    def get_next_batch(self):
        """"""
        start = (self.batch_counter - 1) * self.batch_size
        end = self.batch_counter * self.batch_size
        self.batch_counter = (self.batch_counter + 1) % self.get_number_train_batches()
        return self.train_x[start:end], self.train_y[start:end], self.train_len[start:end]
