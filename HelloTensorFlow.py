import tensorflow as tf

helloConstant = tf.constant("Hello world")
helloVariable = tf.Variable(5)
nFeatures = 120
nLabels = 5
weights = tf.Variable(tf.truncated_normal((nFeatures,nLabels)))
bias = tf.Variable(tf.zeros(nLabels))
init = tf.global_variables_initializer()

with tf.Session() as session:
    session.run(init)
    output = session.run(weights)
    print(output)
    output = session.run(bias)
    print(output)