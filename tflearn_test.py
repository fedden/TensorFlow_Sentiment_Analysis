from __future__ import division, print_function, absolute_import

import tflearn
from utils import *

train, test, valid, stats = get_split_dataset(0.8, 0.1, 0.1)
hidden_parameters = 256

train_examples = [list(column) for column in zip(*[d.values() for d in train])]
trainX = np.array(train_examples[1]).reshape((-1, stats['max_sequence_length']))
trainY = np.array(train_examples[0])

test_examples = [list(column) for column in zip(*[d.values() for d in test])]
testX = np.array(test_examples[1]).reshape((-1, stats['max_sequence_length']))
testY = np.array(test_examples[0])

# Network building
net = tflearn.input_data([None, stats['max_sequence_length']])
# Masking is not required for embedding, sequence length is computed prior to
# the embedding op and assigned as 'seq_length' attribute to the returned Tensor.
net = tflearn.embedding(net, input_dim=(stats['vocabulary_size'] + 1),
                        output_dim=hidden_parameters)
net = tflearn.lstm(net, hidden_parameters, dropout=0.8, dynamic=True)
net = tflearn.fully_connected(net, 5, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0)
model.fit(trainX, trainY, validation_set=(testX, testY), show_metric=True,
          batch_size=32)
