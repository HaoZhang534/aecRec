import tensorflow as tf
import numpy as np
import random as rnd
import math


def to_sparse(indx, rating_matrix, uid_max):
    res = np.zeros([len(indx), uid_max])
    mask = np.zeros([len(indx), uid_max])
    for i in range(len(indx)):
        ri = rating_matrix[indx[i]]
        for pair in ri:
            res[i, pair[0]] = pair[1]
            mask[i, pair[0]] = 1.0
    return res, mask


def test(test_matrix, train_matrix, sess, x, h, mask, uid_max):
    n = 0
    sq_sum = 0
    for key in test_matrix:
        tst_x, tst_mask = to_sparse([key], test_matrix, uid_max)
        tst_x = tst_x.flatten()
        tst_mask = tst_mask.flatten()
        n += np.sum(tst_mask)

        train_x, train_mask = to_sparse([key], train_matrix, uid_max)
        predict = sess.run(h, feed_dict={x: train_x, mask: train_mask})
        predict = predict.flatten()
        d = (tst_x - predict) * tst_mask
        sq_sum += np.sum(d * d)

    rmse = math.sqrt(sq_sum / n)
    print("RMSE: " + str(rmse))
    return rmse


def file2dict(filename):
    rating_matrix = dict()
    min_rating = 100
    tr = open(filename, 'r')

    for line in tr:
        line = line.strip()
        parts = line.split(' ')
        assert parts[2] == parts[3]
        num = int(parts[2])
        uid = int(parts[4].split(':')[0])
        for j in range(num):
            iid = int(parts[4 + num + j].split(':')[0])
            rating = float(parts[4 + num + j].split(':')[1])
            if rating < min_rating:
                min_rating = rating

            if iid in rating_matrix:
                rating_matrix[iid].append([uid, rating])
            else:
                rating_matrix[iid] = [[uid, rating]]
    tr.close()
    return rating_matrix


def main(lamb=0.1, k=500, batch_size=1000, num_steps=200, learning_rate=0.001, display_step=10, test_step=30,uid_max=69878):
    rating_matrix = file2dict("./trainingset")
    test_matrix = file2dict("testingset")
    x = tf.placeholder("float", [None, uid_max])
    mask = tf.placeholder("float", [None, uid_max])

    regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
    g = tf.layers.dense(inputs=x, units=k, activation=tf.nn.sigmoid, kernel_regularizer=regularizer)
    h1 = tf.layers.dense(inputs=g, units=uid_max, activation=tf.nn.relu, kernel_regularizer=regularizer) + 0.5
    c = tf.constant(value=5.0)
    h = tf.minimum(h1, c)
    loss = (tf.reduce_sum(tf.pow((x - h) * mask,2))) / tf.reduce_sum(mask)
    l2_loss = tf.losses.get_regularization_loss()
    loss += l2_loss
    # loss2 = lamb * (tf.reduce_sum(V * V) + tf.reduce_sum(W * W))
    # loss = loss1 + loss2
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(loss)

    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start Training
    # Start a new TF session
    sess = tf.Session()

    # Run the initializer
    sess.run(init)

    # Training
    for i in range(1, num_steps + 1):
        indx = rnd.sample(list(rating_matrix.keys()), batch_size)
        batch_x, batch_mask = to_sparse(indx, rating_matrix, uid_max)

        _, ls, pred, ls2 = sess.run([optimizer, loss, h, l2_loss], feed_dict={x: batch_x, mask: batch_mask})
        if i % display_step == 0 or i == 1:
            # tf.metrics.mean_squared_error()
            print('Step %i: Minibatch Loss: %f' % (i, ls))
            print(str(pred))
            print(str(ls2))

        if i % test_step == 0 or i == 1:
            test(test_matrix, rating_matrix, sess, x, h, mask, uid_max)

    # Testing
    test(test_matrix, rating_matrix, sess, x, h, mask, uid_max)
    sess.close()


if __name__ == '__main__':
    main(lamb=0.00002, batch_size=1800, num_steps=1500,test_step=100)
