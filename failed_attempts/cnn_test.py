from dataset import DataSet
import tensorflow as tf
import datetime
import time
import os

dataset = DataSet(128, 0.8, 0.1, 0.1, False, 8)
max_sequence_length = dataset.stats['max_sequence_length']
number_classes = 5
vocabulary_size = (dataset.stats['vocabulary_size'] + 1)
embedding_size = 300
filter_sizes = [3, 4, 5, 6, 7, 8]
number_filters = 128
learning_rate = 0.05

x = tf.placeholder(tf.int32, [None, max_sequence_length], name="x")
y = tf.placeholder(tf.float32, [None, number_classes], name="y")
dropout_keep_probability = tf.placeholder(tf.float32,
                                          name="dropout_keep_probability")

with tf.name_scope("embedding"):
    embedding_weights = tf.random_uniform([vocabulary_size, embedding_size],
                                          -1.0, 1.0, name="embedding_weights")
    embedded_chars = tf.nn.embedding_lookup(embedding_weights, x)
    embedded_chars_expanded = tf.expand_dims(embedded_chars, -1)

pooled_outputs = []
for i, filter_size in enumerate(filter_sizes):

    with tf.name_scope("convolution-maxpool-%s" % filter_size):

        with tf.name_scope("convolution-layer"):

            filter_shape = [filter_size, embedding_size, 1, number_filters]
            weights = tf.Variable(tf.truncated_normal(filter_shape,
                                                      stddev=0.1),
                                  name="weights")
            biases = tf.Variable(tf.constant(0.1, shape=[number_filters]),
                                 name="biases")
            convolution = tf.nn.conv2d(embedded_chars_expanded, weights,
                                       strides=[1, 1, 1, 1], padding="VALID",
                                       name="convolution")
        with tf.name_scope("non-linearity"):
            activation = tf.nn.relu(tf.nn.bias_add(convolution, biases),
                                    name="relu")
        with tf.name_scope("pooled-layer"):
            size = max_sequence_length - filter_size + 1
            pooled = tf.nn.max_pool(activation,
                                    ksize=[1, size, 1, 1],
                                    strides=[1, 1, 1, 1], padding="VALID",
                                    name="pool")
            pooled_outputs.append(pooled)

with tf.name_scope("concatenating-pooled-features"):
    number_filters_total = number_filters * len(filter_sizes)
    concatenated_pool = tf.concat(pooled_outputs, 3)
    flattened_pool = tf.reshape(concatenated_pool, [-1, number_filters_total])

with tf.name_scope("dropout"):
    dropout = tf.nn.dropout(flattened_pool, dropout_keep_probability)

with tf.name_scope("output"):
    weights_output = tf.Variable(tf.truncated_normal([number_filters_total,
                                                      number_classes],
                                                     stddev=0.1),
                                 name="weights-out")
    biases_output = tf.Variable(tf.constant(0.1, shape=[number_classes]),
                                name="biases-out")
    scores = tf.nn.xw_plus_b(dropout, weights_output, biases_output,
                             name="scores")
    prediction = tf.argmax(scores, 1, name="prediction")

with tf.name_scope("loss"):
    losses = tf.nn.softmax_cross_entropy_with_logits(logits=scores, labels=y)
    loss = tf.reduce_mean(losses)

with tf.name_scope("accurary"):
    correct_predictions = tf.equal(prediction, tf.argmax(y, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32),
                              name="accuracy")

with tf.Session() as sess:
    optimiser = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(loss)
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs",
                                           timestamp))
    loss_summary = tf.summary.scalar("loss", loss)
    accuracy_summary = tf.summary.scalar("accuracy", accuracy)
    train_summary_op = tf.summary.merge([loss_summary, accuracy_summary])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    train_summary_writer = tf.summary.FileWriter(train_summary_dir,
                                                 sess.graph)
    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)

    init_op = tf.global_variables_initializer()
    saver = tf.train.Saver(tf.global_variables())
    step = 0
    while True:

        sess.run(init_op)
        trainX, trainY, _ = dataset.get_next_train_batch()
        feed_dict = {
            x: trainX,
            y: trainY,
            dropout_keep_probability: 0.5
        }

        _, model_summaries, model_loss, model_accuracy = sess.run(
            [optimiser, train_summary_op, loss, accuracy], feed_dict
        )

        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step,
                                                        model_loss,
                                                        model_accuracy))
        train_summary_writer.add_summary(model_summaries, step)

        if step % 10 == 0:
            path = saver.save(sess, checkpoint_prefix,
                              global_step=step)
            print("Saved model checkpoint to {}\n".format(path))
