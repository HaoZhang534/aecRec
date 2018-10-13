import tensorflow as tf
import numpy as np
import random as rnd
import os
import time
import math
import sys
# from scipy.sparse import coo_matrix,csr_matrix

FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("uid_max",69877,"max uid")
tf.flags.DEFINE_integer("item_num",10677,"item num")
tf.flags.DEFINE_integer("test_item_num",9797,"item num")
tf.flags.DEFINE_string("train_file","./i_train","train file")
tf.flags.DEFINE_string("test_file","./i_test","test file")
tf.flags.DEFINE_string("model_path","./model","test file")
tf.flags.DEFINE_integer("gpu_num",4,"gpu number")

def load_from_disk(filename):
    dataset = tf.data.Dataset.from_tensor_slices(filename)
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)))
    return dataset


def parse_py_fn(line,mode):
    uid_max=FLAGS.uid_max
    if mode.startswith('train'):
        ratings=np.zeros([uid_max+1])
        pairs=line.split()
        pairs.pop(0)
        pairs.pop(0)
        for pair in pairs:
            pair=pair.decode().split(":")
            ratings[int(pair[0])]=float(pair[1])
        return ratings.astype(np.float32)
    else:
        assert mode.startswith('test')
        tr_rat=np.zeros([uid_max+1])
        tst_rat=np.zeros([uid_max+1])
        pairs=line.split()
        pairs.pop(0)
        tst_num=int(pairs.pop(0))
        for i in range(tst_num):
            pair=pairs.pop(0).decode().split(':')
            tst_rat[int(pair[0])]=float(pair[1])
        tr_num=int(pairs.pop(0))
        for i in range(tr_num):
            pair=pairs.pop(0).decode().split(':')
            tr_rat[int(pair[0])]=float(pair[1])
        return tr_rat.astype(np.float32),tst_rat.astype(np.float32)


def input(dataset,mode,params):
    uid_max=FLAGS.uid_max
    parse_py_fn_=lambda line:parse_py_fn(line,mode)
    if mode.startswith('train'):
        dataset=dataset.map(lambda line:tuple(tf.py_func(parse_py_fn_,[line],[tf.float32])))
    else:
        dataset=dataset.map(lambda line:tuple(tf.py_func(parse_py_fn_,[line],[tf.float32,tf.float32])))

    dataset=dataset.cache()
    if mode.startswith('train'):
        dataset = dataset.shuffle(buffer_size=FLAGS.item_num)
    if mode=='train_multi':
        assert  params['batch_size']%FLAGS.gpu_num == 0,"batch_size should be divided evenly by gpu_num"
        d_r=True
    else:
        d_r=False
    dataset=dataset.batch(params['batch_size'],drop_remainder=d_r)
    if mode.startswith('train'):
        dataset=dataset.repeat(params['repeat_times'])
    iterator = dataset.make_one_shot_iterator()

    if mode.startswith('train'):
        mat,=iterator.get_next()
        return mat
    else:
        mat1,mat2=iterator.get_next()
        return mat1,mat2


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


def run(filename,mode,batch_size=1000,repeat_times=100,test_ep=10,save_ep=50,lamb=0.00001,sess=None,restore_ep=0):
    dataset=load_from_disk([filename])

    params={"batch_size":batch_size,"repeat_times":repeat_times}

    # sess.run(mat)
    if mode.startswith("train"):
        mat = input(dataset, mode, params)
        # op,grads=model(mat,mode,lamb=lamb)
        if mode=='train_multi':
            opt=model(mat,mode,lamb=lamb)
        else:
            met_opt,opt=model(mat,mode,lamb=lamb)
        # gd=[]
        # var=[]
        # for pair in(grads):
        #     gd.append(pair[0])
        #     var.append(pair[1])
        tr_metric_init_op = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="AE_TR_METRIC"))
    else:
        assert mode.startswith('test')
        mat1,mat2=input(dataset,mode,params)
        op=model(tr_mat=mat1,mode=mode,lamb=lamb,tst_mat=mat2)
        tr_metric_init_op = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="AE_TST_METRIC"))


    if sess==None:
        config = tf.ConfigProto(
            allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)



    if mode.startswith("train"):
        num_steps = math.ceil(FLAGS.item_num / batch_size)
        saver=tf.train.Saver()
        if restore_ep==0:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess,"%s/ae%s%d"%(FLAGS.model_path,mode,restore_ep))
        # Training
        ti=time.time()
        for epoch in range(1,repeat_times+1):
            sess.run(tr_metric_init_op)
            for i in range(1, num_steps + 1):
                print("epoch:%d,step:%d   " % (restore_ep + epoch, i))
                if mode=='train_multi':
                    _=sess.run([opt])
                else:
                    metric,_=sess.run([met_opt,opt])
                    print(str(metric))

                # for j1,j2 in zip(gdval,var):
                #     print(str(j1)+'hh')
                #     print(str(j2))
            dur=time.time()-ti
            ti=time.time()
            print("time:"+str(dur))

            if (epoch+restore_ep)%save_ep==0 or epoch==repeat_times:
                save_path="%s/ae%s%d"%(FLAGS.model_path,mode,epoch+restore_ep)
                saver.save(sess,save_path)
                print("Model Saved!")
            if (epoch+restore_ep) % test_ep == 0 or epoch == repeat_times:
                run(FLAGS.test_file,'test',sess=sess,batch_size=2000)


    if mode.startswith("test"):
        num_steps = math.ceil(FLAGS.test_item_num / batch_size)
        sess.run(tr_metric_init_op)
        for i in range(1,num_steps+1):
            metric=sess.run(op)
            print("MSRE:"+str(metric))

def model(tr_mat,mode,lamb=0.00001, k=500,learning_rate=0.001,tst_mat=None):
    uid_max = FLAGS.uid_max+1
    tr_mask = tf.sign(tr_mat)
    regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)
    if mode!='train_multi':
        with tf.variable_scope("AE",reuse=tf.AUTO_REUSE):
            v = tf.get_variable("v", shape=[uid_max, k], regularizer=regularizer)
            w = tf.get_variable("w", shape=[k, uid_max], regularizer=regularizer)
            b1 = tf.get_variable("b1", shape=[k], initializer=tf.zeros_initializer)
            b2 = tf.get_variable("b2", shape=[uid_max], initializer=tf.zeros_initializer)
            pre1 = tf.nn.bias_add(tf.matmul(tr_mat, v), b1)
            g = tf.nn.sigmoid(pre1)

            pre2 = tf.nn.bias_add(tf.matmul(g, w), b2)
            h1 = tf.nn.relu(pre2) + 0.5
            c = tf.constant(value=5.0)
            h = tf.minimum(h1, c)
    else:
        tower_grads = []
        optimizer = tf.train.AdamOptimizer(learning_rate)
        xs=tf.split(tr_mat,FLAGS.gpu_num,0)
        masks=tf.split(tr_mask,FLAGS.gpu_num,0)
        with tf.variable_scope("AE"):  # ,reuse=tf.AUTO_REUSE
            for gpu_indx in range(FLAGS.gpu_num):
                with tf.device('/gpu:%d' % gpu_indx):
                    with tf.name_scope('%s_%d' % ("tower", gpu_indx)) as scope:
                        v = tf.get_variable("v", shape=[uid_max, k], regularizer=regularizer)
                        w = tf.get_variable("w", shape=[k, uid_max], regularizer=regularizer)
                        b1 = tf.get_variable("b1", shape=[k], initializer=tf.zeros_initializer)
                        b2 = tf.get_variable("b2", shape=[uid_max], initializer=tf.zeros_initializer)
                        pre1 = tf.nn.bias_add(tf.matmul(xs[gpu_indx], v), b1)
                        g = tf.nn.sigmoid(pre1)
                        pre2 = tf.nn.bias_add(tf.matmul(g, w), b2)
                        h1 = tf.nn.relu(pre2) + 0.5
                        c = tf.constant(value=5.0)
                        h = tf.minimum(h1, c)
                        loss = (tf.reduce_sum(tf.pow((xs[gpu_indx] - h) * masks[gpu_indx], 2))) / tf.reduce_sum(
                            masks[gpu_indx])
                        l2_loss = tf.losses.get_regularization_loss()
                        loss += l2_loss
                        tf.get_variable_scope().reuse_variables()
                        tower_grads.append(optimizer.compute_gradients(loss))
        grads = average_gradients(tower_grads)
        apply_gradient_op = optimizer.apply_gradients(grads)
        return apply_gradient_op

    if mode.startswith("train"):
        with tf.variable_scope("AE_TR_METRIC"):
            tr_metric = {
                "RMSE": tf.metrics.root_mean_squared_error(tr_mat, h, weights=tr_mask)}
            tr_metric_op = tf.tuple([op for _, op in tr_metric.values()])
        loss=tf.losses.mean_squared_error(tr_mat,h,weights=tr_mask)+tf.losses.get_regularization_loss()
        opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
        return tr_metric_op,opt
    else:
        assert tst_mat != None
        tst_mask=tf.sign(tst_mat)
        with tf.variable_scope('AE_TST_METRIC'):
            tst_metric = {
                "RMSE": tf.metrics.root_mean_squared_error(tst_mat, h, weights=tst_mask)}
            tr_metric_op = tf.tuple([op for _, op in tst_metric.values()])

        return tr_metric_op

if __name__ == '__main__':
    if len(sys.argv)==1:
        restore_ep=0
    else:
        restore_ep=int(sys.argv[1])
    run(FLAGS.train_file,"train_multi",1800,500,test_ep=50,restore_ep=restore_ep,lamb=0.00003,save_ep=30)
