from __future__ import division, print_function, absolute_import

import tflearn
from dataset import *

dataset = DataSet(32, 0.8, 0.1, 0.1, False, 8)

hidden_parameters = 256

trainX, trainY, _ = dataset.get_train_set()
validX, validY, _ = dataset.get_valid_set()

net = tflearn.input_data([None, dataset.stats['max_sequence_length']])
net = tflearn.embedding(net, input_dim=(dataset.stats['vocabulary_size'] + 1),
                        output_dim=hidden_parameters)
net = tflearn.lstm(net, hidden_parameters, dropout=0.8, dynamic=True)
net = tflearn.fully_connected(net, 5, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=0.001,
                         loss='categorical_crossentropy')

# Training
model = tflearn.DNN(net, tensorboard_verbose=0, checkpoint_path="checkpoints/tflearn_0")
model.fit(trainX, trainY, validation_set=(validX, validY), show_metric=True,
          batch_size=32)
