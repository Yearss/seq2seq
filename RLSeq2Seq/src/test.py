
import tensorflow as tf


# FLAGS = tf.app.flags.FLAGS
#
# tf.app.flags.DEFINE_string("test_param", "hansongbo", "testing.")
#
# print FLAGS.test_param
#
# flags = getattr(FLAGS, "__flags")
#
# print(flags)

q = tf.FIFOQueue(10, "float")

counter = tf.Variable(0.0)

increment_op = tf.assign_add(counter, 1.0)

enqueue_op = q.enqueue(counter)

qr = tf.train.QueueRunner(q, enqueue_ops=[increment_op, enqueue_op] * 2)

with tf.Session() as sess:

    sess.run(tf.global_variables_initializer())

    # build coord
    coord = tf.train.Coordinator()

    enque_threads = qr.create_threads(sess, coord=coord, start=True)

    while True:
        output = sess.run(q.dequeue())
        print(output)
        if output > 1000:
            coord.request_stop()
            break

    coord.join(enque_threads)



if __name__ == "__main":

    tf.app.run()