import tensorflow as tf
tf.reset_default_graph()   # To clear the defined variables and operations of the previous cell
# create graph
a = tf.constant(2)
b = tf.constant(3)
c = tf.add(a, b)
# creating the writer out of the session
# writer = tf.summary.FileWriter('./graphs', tf.get_default_graph())
# launch the graph in a session
with tf.Session() as sess:
    # or creating the writer inside the session
    writer = tf.summary.FileWriter('./graphs', sess.graph)
    print(sess.run(c))
