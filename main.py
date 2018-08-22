
import tensorflow as tf


def main():
    # Build your graph.
    with tf.Session() as sess:
        x = tf.constant([[37.0, -23.0], [1.0, 4.0]])
        w = tf.Variable(tf.ones([2, 2]))
        y = tf.matmul(x, w)
        # ...
        sess.run(tf.global_variables_initializer())
        print(sess.run(x))
        print(sess.run(w))
        print(sess.run(y))

        loss = tf.reduce_sum(tf.square(tf.subtract(y, [[1, 0], [0, 1]])))
        train_op = tf.train.AdamOptimizer(0.01).minimize(loss)

        # `sess.graph` provides access to the graph used in a <a href="../api_docs/python/tf/Session"><code>tf.Session</code></a>.
        writer = tf.summary.FileWriter("/tmp/log/bbbb", sess.graph)


        # Perform your computation...
        for i in range(10000):
            sess.run(train_op)
            if i % 1000 == 0:
                print(sess.run(loss))

        print(sess.run(x))
        print(sess.run(w))
        print(sess.run(tf.matmul(x, w)))
        writer.close()

        print(sess.graph.get_all_collection_keys())
        vars = sess.graph.get_collection('variables')
        pass


        # builder = tf.saved_model.builder.SavedModelBuilder('/tmp/log/bbbb/savedbuilder')
        # builder.add_meta_graph_and_variables(sess, ['tag_a'])
        # builder.save(True)


main()
