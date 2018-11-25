# Solution is available in the other "solution.py" tab
import tensorflow as tf

softmax_data = [0.7, 0.2, 0.1]
one_hot_data = [1.0, 0.0, 0.0]

softmax = tf.placeholder(tf.float32)
one_hot = tf.placeholder(tf.float32)

# crossEntropy = -tf.reduce_sum(tf.multiply(tf.log(softmax),one_hot))

# TODO: Print cross entropy from session
with tf.Session() as sess:
    softMaxValue = sess.run(softmax,feed_dict={softmax:softmax_data})
    softMaxValue = sess.run(tf.log(softMaxValue))
    oneHotValue = sess.run(one_hot,feed_dict={one_hot:one_hot_data})
    print(softMaxValue)
    print(oneHotValue)
    ylny_ = sess.run(tf.multiply(softMaxValue,oneHotValue))
    crossEntropy = sess.run(-tf.reduce_sum(ylny_))
    # crossEntropy = sess.run(crossEntropy,feed_dict={softmax:softmax_data,one_hot:one_hot_data})
    print(crossEntropy)
