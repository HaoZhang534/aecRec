import tensorflow as tf
import numpy as np
import random as rnd
import os
import time

def to_sparse(indx, rating_matrix, uid_max):
    res = np.zeros([len(indx), uid_max])
    mask = np.zeros([len(indx), uid_max])
    for i in range(len(indx)):
        ri = rating_matrix[indx[i]]
        for pair in ri:
            res[i, pair[0]] = pair[1]
            mask[i, pair[0]] = 1.0
    return res, mask

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


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.

    Args:
      tower_grads: List of lists of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, _ in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            if g is None:
                continue

            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat(axis=0, values=grads)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads


def main(lamb=0.1, k=500, batch_size=1000, num_steps=200, learning_rate=0.001, display_step=10, test_step=30,
         uid_max=69878, gpu_num=4, save_step=100):
    rating_matrix = file2dict("./trainingset")
    test_matrix = file2dict("testingset")
    assert batch_size % gpu_num == 0, "batch_size % gpu_num != 0"
    bat_per_gpu = batch_size / gpu_num
    optimizer = tf.train.AdamOptimizer(learning_rate)
    x = tf.placeholder("float", [batch_size, uid_max])
    mask = tf.placeholder("float", [batch_size, uid_max])

    xs = tf.split(x, gpu_num, 0)
    masks = tf.split(mask, gpu_num, 0)
    tower_grads = []
    regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
    with tf.variable_scope(tf.get_variable_scope()):
        for gpu_indx in range(gpu_num):
            with tf.device('/gpu:%d' % gpu_indx):
                with tf.name_scope('%s_%d' % ("tower", gpu_indx)) as scope:
                    v = tf.get_variable("v", shape=[uid_max, k], regularizer=regularizer)
                    w = tf.get_variable("w", shape=[k, uid_max], regularizer=regularizer)
                    b1 = tf.get_variable("b1", shape=[k],initializer=tf.zeros_initializer)
                    b2 = tf.get_variable("b2", shape=[uid_max], initializer=tf.zeros_initializer)
                    pre1 = tf.nn.bias_add(tf.matmul(xs[gpu_indx], v), b1)
                    g = tf.nn.sigmoid(pre1)

                    pre2 = tf.nn.bias_add(tf.matmul(g, w), b2)
                    h1 = tf.nn.relu(pre2) + 0.5
                    c = tf.constant(value=5.0)
                    h = tf.minimum(h1, c)
                    loss = (tf.reduce_sum(tf.pow((xs[gpu_indx] - h) * masks[gpu_indx], 2))) / tf.reduce_sum(masks[gpu_indx])
                    tf.get_variable_scope().reuse_variables()
                    tower_grads.append(optimizer.compute_gradients(loss))

                    if (gpu_indx == 0):
                        l2_loss = tf.losses.get_regularization_loss()
                        tower_grads.append(optimizer.compute_gradients(l2_loss))

    grads = average_gradients(tower_grads)
    apply_gradient_op = optimizer.apply_gradients(grads)
    # optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver(tf.global_variables())
    # Initialize the variables (i.e. assign their default value)
    init = tf.global_variables_initializer()

    # Start Training
    # Start a new TF session
    config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=True)
    config.gpu_options.allow_growth = True

    sess = tf.Session(config=config)

    # Run the initializer
    sess.run(init)

    # Training
    last_t=time.time()
    for i in range(1, num_steps + 1):
        indx = rnd.sample(list(rating_matrix.keys()), batch_size)
        batch_x, batch_mask = to_sparse(indx, rating_matrix, uid_max)

        _, pred, ls = sess.run([apply_gradient_op, h, loss], feed_dict={x: batch_x, mask: batch_mask})
        if i % display_step == 0 or i == 1:
            dur=time.time()-last_t
            last_t=time.time()
            # tf.metrics.mean_squared_error()
            print('Step %i: time consumption:%f Minibatch Loss: %f' % (i, dur, ls))
            print(str(pred))

        if i % save_step == 0 or i==num_steps :
            checkpoint_path = os.path.join('./model', 'model.ckpt')
            saver.save(sess, checkpoint_path, global_step=i)

        # if i % test_step == 0 or i == 1:
        #     test(test_matrix, rating_matrix, sess, x, h, mask, uid_max)

    # Testing
    # test(test_matrix, rating_matrix, sess, x, h, mask, uid_max)
    sess.close()

if __name__ == '__main__':
    main(lamb=0.02, batch_size=1800, num_steps=1500, test_step=100,save_step=1)
