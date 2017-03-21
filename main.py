from dataset import DataSet
import tensorflow as tf
from model import *
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--train", help="train the model", action="store_true")
parser.add_argument("--test", help="test the model", action="store_true")
args = parser.parse_args()

mode_train = False if args.test else True
mode_train = True if args.train else False
seed = 8
batch_size = 64
dataset = DataSet(batch_size, 0.8, 0.1, 0.1, seed)
learning_rate = 0.001
training_iterations = 50
display_step = 10
embedding_size = 128
sequence_max_length = dataset.stats['max_sequence_length']
number_hidden_params = 128
number_classes = 5
checkpoint_file_name = "checkpoints/lstm_tf_sa.ckpt"

weights, biases = model_weights(dataset, embedding_size, seed,
                                number_hidden_params, number_classes)

sequence_length = tf.placeholder(tf.int32, [None])
x = tf.placeholder(tf.float32, [None, sequence_max_length, 1])
y = tf.placeholder(tf.float32, [None, number_classes])

embedding = word_embedding(x, weights, sequence_max_length, embedding_size)
pred = dynamic_lstm(embedding, sequence_length, sequence_max_length, weights,
                    biases, number_hidden_params, embedding_size)
cost = model_cost(y, pred)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)
accuracy = model_accuracy(y, pred)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

with tf.Session() as sess:

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('logs/train', sess.graph)

    if mode_train:
        sess.run(init)
    else:
        saver.restore(sess, checkpoint_file_name)

    v_x, v_y, v_l = dataset.get_valid_set()
    t_x, t_y, t_l = dataset.get_test_set()

    if mode_train:
        print "Training."
        global_step = 0
        best_accuracy = 0
        for e in range(10000):

            print "\nEpoch: " + str(e)
            step = 1
            while (step - 1) < dataset.get_number_train_batches():

                b_x, b_y, b_l = dataset.get_next_train_batch()
                sess.run(optimizer, feed_dict={x: b_x, y: b_y,
                                               sequence_length: b_l})

                if step % display_step == 0:

                    # Calculate batch accuracy
                    acc = sess.run(accuracy, feed_dict={x: v_x, y: v_y,
                                                        sequence_length: v_l})

                    # A form of early stopping in a way - perhaps should be
                    # called early saving...
                    if (acc > best_accuracy):
                        best_accuracy = acc
                        save_path = saver.save(sess, checkpoint_file_name)
                        print("New best validation accuracy, saved in file: %s"
                              % save_path)

                    # Calculate batch loss
                    loss, summary = sess.run([cost, merged],
                                             feed_dict={x: v_x,
                                                        y: v_y,
                                                        sequence_length: v_l})
                    print("Current batch loss: " +
                          "{:.6f}".format(loss) + ", Accuracy: " +
                          "{:.5f}".format(acc))
                    train_writer.add_summary(summary, global_step)

                global_step += 1
                step += 1
        print "Optimization Finished!"

    print "Testing."
    acc = sess.run(accuracy, feed_dict={x: t_x, y: t_y, sequence_length: t_l})
    loss = sess.run(cost, feed_dict={x: t_x, y: t_y, sequence_length: t_l})
    print("Testing loss: " + "{:.6f}".format(loss) +
          ", Accuracy: " + "{:.5f}".format(acc))
print "Done!"
