import tensorflow as tf

def variable_summaries(var):
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization).

    https://www.tensorflow.org/get_started/summaries_and_tensorboard
    """
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
        with tf.name_scope('stddev'):
            stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
            tf.summary.scalar('stddev', stddev)
            tf.summary.scalar('max', tf.reduce_max(var))
            tf.summary.scalar('min', tf.reduce_min(var))
            tf.summary.histogram('histogram', var)

def word_embedding(x, weights, dimension=3):
    """Inspired from: http://stackoverflow.com/a/35296384
    """
    with tf.name_scope("Word_Embedding"):
        embedding_layer = tf.nn.embedding_lookup(weights['embedding'],
                                                 tf.cast(x, tf.int32))
        variable_summaries(weights['embedding'])
        return tf.reduce_sum(embedding_layer, [dimension])

def dynamic_lstm(x, seqlen, sequence_max_length, weights, biases, number_hidden_params):
    """Returns a LSTM tf model.

    First the tensors are reshaped from (batch_size, n_steps, n_input) to
    'n_steps' list of shape (batch_size, n_input). The tensorts are next past
    into the lstm.
    """
    with tf.name_scope("Reshaping_Tensors"):
        x = tf.transpose(x, [1, 0, 2])
        x = tf.reshape(x, [-1, 1])
        x = tf.split(axis=0, num_or_size_splits=sequence_max_length, value=x)

    with tf.name_scope("LSTM"):
        lstm_cell = tf.contrib.rnn.BasicLSTMCell(number_hidden_params)
        outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x, dtype=tf.float32,
                                                    sequence_length=seqlen)
    with tf.name_scope("Get_Last_LSTM_Output"):
        outputs = tf.stack(outputs)
        outputs = tf.transpose(outputs, [1, 0, 2])
        # Hack to build the indexing and retrieve the right output.
        batch_size = tf.shape(outputs)[0]
        # Start indices for each sample
        index = tf.range(0, batch_size) * sequence_max_length + (seqlen - 1)
        # Indexing
        outputs = tf.gather(tf.reshape(outputs, [-1, number_hidden_params]), index)

    with tf.name_scope("Linear_Activation"):
        linear_activation = tf.matmul(outputs, weights['out']) + biases['out']
        variable_summaries(weights['out'])
        variable_summaries(biases['out'])
        return linear_activation

def model_cost(actual, prediction):
    """Returns the cost."""
    with tf.name_scope("Cost"):
        cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,
                                                                      labels=actual))
        tf.summary.scalar('cost', cost)
    return cost

def model_accuracy(actual, prediction):
    """Returns accuracy of predictions."""
    with tf.name_scope("Accuracy"):
        with tf.name_scope("Correct_Prediction"):
            correct_pred = tf.equal(tf.argmax(prediction, 1), tf.argmax(actual, 1))
        with tf.name_scope("Accuracy"):
            accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
            tf.summary.scalar('accuracy', accuracy)
    return accuracy
