from model import *
from dataset import DataSet
import tensorflow as tf

seed = 8
batch_size = 32
dataset = DataSet(batch_size, 0.8, 0.1, 0.1, seed)
learning_rate = 0.001
training_iterations = 1000000
display_step = 10
embedding_size = 128
sequence_max_length = dataset.stats['max_sequence_length']
number_hidden_params = 256
number_classes = 5

print "Stats Dictionary:"
print dataset.stats

weights = {
    'embedding' : tf.Variable(tf.truncated_normal([(dataset.stats['vocabulary_size'] + 1),
                                                  embedding_size], stddev=0.01,
                                                  seed=seed)),
    'out'       : tf.Variable(tf.truncated_normal([number_hidden_params, number_classes],
                                                  stddev=0.01, seed=seed))
}

biases = {
    'out' : tf.Variable(tf.truncated_normal([number_classes], stddev=0.01,
                                            seed=seed))
}

sequence_length = tf.placeholder(tf.int32, [None])
x = tf.placeholder(tf.float32, [None, sequence_max_length, 1])
y = tf.placeholder(tf.float32, [None, number_classes])

embedding = word_embedding(x, weights)
pred = dynamic_lstm(embedding, sequence_length, sequence_max_length, weights,
                    biases, number_hidden_params)
cost = model_cost(y, pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = model_accuracy(y, pred)
init = tf.global_variables_initializer()

with tf.Session() as sess:

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)

    print "Initialising variables."
    sess.run(init)
    # Keep training until reach max iterations
    for e in range(100):
        print "\nEpoch: " + str(e)
        step = 1
        while (step - 1) < dataset.get_number_train_batches():

            b_x, b_y, b_l = dataset.get_next_batch()
            sess.run(optimizer, feed_dict={x: b_x, y: b_y, sequence_length: b_l})

            if step % display_step == 0:
                # Calculate batch accuracy
                acc = sess.run(accuracy, feed_dict={x: b_x, y: b_y,
                                                    sequence_length: b_l})
                # Calculate batch loss
                loss, summary = sess.run([cost, summary], feed_dict={x: b_x, y: b_y,
                                                                     sequence_length: b_l})
                print("Iter " + str(step*batch_size) + ", Minibatch Loss= " + \
                      "{:.6f}".format(loss) + ", Training Accuracy= " + \
                      "{:.5f}".format(acc))
                train_writer.add_summary(summary, i)

            step += 1
    print "Optimization Finished!"
print "Done!"
