import tensorflow as tf
import numpy as np
import random as rnd
import os
import time
import math
import sys
from tensorflow.data import Dataset as dt
# from scipy.sparse import coo_matrix,csr_matrix
from multiprocessing import Pool


FLAGS = tf.flags.FLAGS
tf.flags.DEFINE_integer("uid_max",69877,"max uid")
tf.flags.DEFINE_integer("item_num",10677,"item num")
# tf.flags.DEFINE_integer("test_item_num",9797,"item num")
tf.flags.DEFINE_integer("genre_num",20,"item num")
tf.flags.DEFINE_string("train_file","./i_train","train file")
tf.flags.DEFINE_string("test_file","./i_test","test file")
tf.flags.DEFINE_string("model_path","./model","test file")
tf.flags.DEFINE_string("np_file","./extern.npz","test file")
tf.flags.DEFINE_string("summary_dir","./summary","summary directory")



def model_genre1(tr_mat, mode, params, tst_mat=None, genre=None):

    lamb=params['lamb']
    k1=params['k1']
    k2=params['k2']
    learning_rate=params['lr']
    usr_num = FLAGS.uid_max+1

    tr_mask = tf.sign(tr_mat)
    regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)

    with tf.variable_scope("AE",reuse=tf.AUTO_REUSE):

        g=dense(tr_mat=tr_mat,in_dim=usr_num,out_dim=k1,activation=tf.nn.sigmoid,bias=True,kernel_reg=regularizer,name='ratings_to_hidden')

        g_=tf.concat([g,genre],axis=1)

        feature=dense(g_,k1+FLAGS.genre_num,k2,tf.nn.tanh,True,kernel_reg=regularizer,name='concat_to_feature')
        gate=dense(g_,k1+FLAGS.genre_num,k2,tf.nn.sigmoid,True,kernel_reg=regularizer,name='concat_to_gate')

        ft_gt=feature*gate

        recover=dense(ft_gt,k2,usr_num,tf.nn.relu,True,kernel_reg=regularizer,name='recover_from_gated_feature')

        h1 = tf.nn.relu(recover) + 0.5
        c = tf.constant(value=5.0)
        h = tf.minimum(h1, c)

    if mode.startswith("train"):
        with tf.name_scope("train"):
            with tf.variable_scope("AE_TR_METRIC"):
                tr_metric = {
                    "RMSE": tf.metrics.root_mean_squared_error(tr_mat, h, weights=tr_mask)}
                tr_metric_op = tf.tuple([op for _, op in tr_metric.values()])
                mtc=tf.summary.scalar('RMSE',tr_metric['RMSE'][1])
            loss=tf.losses.mean_squared_error(tr_mat,h,weights=tr_mask)+tf.losses.get_regularization_loss()
            opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)
            ls=tf.summary.scalar('loss',loss)

        # tr_summarys = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='train')
        tr_summary_op = tf.summary.merge([ls,mtc])


        return tr_metric_op,opt,tr_summary_op
    else:
        with tf.name_scope("test"):
            assert tst_mat != None
            tst_mask=tf.sign(tst_mat)
            with tf.variable_scope('AE_TST_METRIC'):
                tst_metric = {
                    "RMSE": tf.metrics.root_mean_squared_error(tst_mat, h, weights=tst_mask)}
                mtc=tf.summary.scalar('RMSE_TST', tst_metric["RMSE"][1])
                tr_metric_op = tf.tuple([op for _, op in tst_metric.values()])

        tst_summary_op=tf.summary.merge([mtc])
        return tr_metric_op,tst_summary_op

def load_from_disk(filename):
    dataset = dt.from_tensor_slices(filename)
    dataset = dataset.flat_map(
        lambda filename: (
            tf.data.TextLineDataset(filename)))
    return dataset

def load_genre_from_np(filename):
    np_file=np.load(filename)
    genre=np_file['gnr_emb']
    genre_dt=dt.from_tensor_slices(genre)
    return genre_dt

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

def np_genre_to_nums():
    np_file=np.load('./extern.npz')
    genre=np_file['gnr_emb']
    print(str(genre.shape))
    genre_lst=[]
    assert genre.shape[0]==FLAGS.item_num
    for i in range(FLAGS.item_num):
        tmp=set()
        for j in range(genre.shape[1]):
            if genre[i][j]==1:
                tmp.add(j)
        genre_lst.append(tmp)

    return genre_lst

def input(dataset,mode,params,genre=None):
    uid_max=FLAGS.uid_max
    parse_py_fn_=lambda line:parse_py_fn(line,mode)
    if mode.startswith('train'):
        dataset=dataset.map(lambda line:tf.py_func(parse_py_fn_,[line],tf.float32))
    else:
        dataset=dataset.map(lambda line:tf.py_func(parse_py_fn_,[line],[tf.float32,tf.float32]))

    if mode.endswith("genre"):
        dataset=dt.zip((dataset,genre))

    dataset=dataset.cache()
    if mode.startswith('train'):
        dataset = dataset.shuffle(buffer_size=100 * params['batch_size'])
    dataset=dataset.batch(params['batch_size'])
    if mode.startswith('train'):
        dataset=dataset.repeat(params['repeat_times'])
    else:
        dataset = dataset.repeat()
        pass
    iterator = dataset.make_one_shot_iterator()
    if mode.startswith('train'):
        if mode.endswith("genre"):
            mat,genre_mat=iterator.get_next()
            return mat,genre_mat
        else:
            mat=iterator.get_next()
            return mat
    else:
        if mode.endswith("genre"):
            mat,genre_mat=iterator.get_next()
            mat1,mat2=mat
            return mat1,mat2,genre_mat
        else:
            mat1, mat2=iterator.get_next()
            return mat1,mat2

def run_genre(filename,np_file_name,mode,model_params,batch_size=1000,repeat_times=100,test_ep=10,save_ep=50,summary_ep=3,sess=None,restore_ep=0,summary_dir=None,summary_writer=None,model_path=None,result_path=None,gpu_indx=0):
    time.sleep(3)
    dt1=load_from_disk([filename])
    genre=load_genre_from_np(np_file_name)
    params = {"batch_size": batch_size, "repeat_times": repeat_times}

    if mode.startswith('train'):
        mat,genre=input(dataset=dt1,mode=mode,params=params,genre=genre)
        # print(sess.run(genre))
        met_opt, opt,lr_ep,add_global=model_genre2(mat, params=model_params, mode=mode, genre=genre,gpu_indx=gpu_indx,epoch=restore_ep)
        tr_metric_init_op = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="train/AE_TR_METRIC"))
    else:
        mat1, mat2, genre = input(dataset=dt1, mode=mode, params=params, genre=genre)
        op=model_genre2(tr_mat=mat1, tst_mat=mat2, params=model_params, mode=mode, genre=genre,gpu_indx=gpu_indx)
        tr_metric_init_op = tf.variables_initializer(
            tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES, scope="test/AE_TST_METRIC"))

    if sess==None:
        config = tf.ConfigProto(
            allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)

    if mode.startswith("train"):
        f_res=open(result_path+"/result","a")
        f_res.write(construct_hp_str(model_params)+'bt_sz:%d'%batch_size+'\n')
        RMSE_list=[]
        test_params=None
        test_batch=2000
        num_steps = math.ceil(FLAGS.item_num / batch_size)
        tst_num_steps = math.ceil(FLAGS.item_num / test_batch)
        saver=tf.train.Saver()
        assert model_path!= None
        if restore_ep==0:
            init = tf.global_variables_initializer()
            sess.run(init)
        else:
            saver.restore(sess,"%s/ae%d"%(model_path,restore_ep))
        # Training
        ti=time.time()
        for epoch in range(1,repeat_times+1):

            sess.run(tr_metric_init_op)
            sess.run(add_global)

            for i in range(1, num_steps + 1):
                metric,_=sess.run([met_opt,opt])
                print("epoch:%d,step:%d   "%(restore_ep+epoch,i))
                print(str(metric))

            dur=time.time()-ti
            ti=time.time()
            print("time:"+str(dur))


            if (epoch+restore_ep)%save_ep==0 or epoch==repeat_times:
                save_path="%s/ae%d"%(model_path,epoch+restore_ep)
                saver.save(sess,save_path)
                print("Model Saved!")
            if (epoch+restore_ep) % test_ep == 0 or epoch == repeat_times:
                if test_params==None:
                    test_params,metric=run_genre(FLAGS.test_file,FLAGS.np_file,model_params=model_params,mode='test_genre',sess=sess,batch_size=test_batch,summary_writer=summary_writer,gpu_indx=gpu_indx)
                    RMSE_list.append(metric)
                    f_res.write("%d     %f\n" % (epoch+restore_ep, metric))
                else:
                    tr_metric_init_op, op=test_params
                    sess.run(tr_metric_init_op)
                    for i in range(1, tst_num_steps + 1):
                        metric, = sess.run(op)
                        print("MSRE:" + str(metric))
                    if len(RMSE_list)>=2 and metric>RMSE_list[len(RMSE_list)-2]:
                        break
                    RMSE_list.append(metric)
                    f_res.write("%d     %f\n" % (epoch+restore_ep, metric))

        f_res.close()
        print("Best performence is %f"%min(RMSE_list))
        return min(RMSE_list)

    if mode.startswith("test"):

        tst_num_steps = math.ceil(FLAGS.item_num / batch_size)
        sess.run(tr_metric_init_op)
        for i in range(1,tst_num_steps+1):
            metric,=sess.run(op)
            print("MSRE:"+str(metric))

        return (tr_metric_init_op,op),metric


def dense(tr_mat,in_dim,out_dim,activation,bias,kernel_reg,name):
    with tf.variable_scope(name, reuse=tf.AUTO_REUSE):
        w=tf.get_variable("w",shape=[in_dim,out_dim],regularizer=kernel_reg)
        if bias:
            b = tf.get_variable("b", shape=[out_dim], initializer=tf.zeros_initializer)
            pre = tf.nn.bias_add(tf.matmul(tr_mat, w), b)
        else:
            pre = tf.matmul(tr_mat, w)

        h=activation(pre)

        return h


def construct_hp_str(params):
    rat_emb_dim = params['rat_emb_dim']
    genre_emb_dim = params['genre_emb_dim']
    feature_dim = params['feature_dim']
    lamb=params['lamb']
    lr=params['lr']
    # drop_out=params['drop_out']
    return 'r%d_g%d_f%d_lb%f_lr_%f'%(rat_emb_dim,genre_emb_dim,feature_dim,lamb,lr)


def model_genre2(tr_mat, mode, params, tst_mat=None, genre=None,gpu_indx=0,epoch=0,lr_decay=False):

    lamb=params['lamb']
    drop_out = params['drop_out']
    last_layer_rglz = params['last_layer_rglz']
    rat_input_dim=FLAGS.uid_max+1
    genre_input_dim=FLAGS.genre_num

    recover_dim=rat_input_dim
    rat_emb_dim=params['rat_emb_dim']
    genre_emb_dim=params['genre_emb_dim']
    feature_dim=params['feature_dim']

    tr_mask = tf.sign(tr_mat)
    # with tf.device('/gpu:%d' % 1):
    with tf.variable_scope("AE",reuse=tf.AUTO_REUSE):
        if last_layer_rglz:
            regularizer=None
        else:
            regularizer=tf.contrib.layers.l2_regularizer(scale=lamb)
        rat_emb=dense(tr_mat=tr_mat,in_dim=rat_input_dim,out_dim=rat_emb_dim,activation=tf.nn.sigmoid,bias=True,kernel_reg=regularizer,name='rat_emb')
        genre_emb=dense(tr_mat=genre,in_dim=genre_input_dim,out_dim=genre_emb_dim,activation=tf.nn.sigmoid,bias=True,kernel_reg=regularizer,name='genre_emb')

        rat_feature=dense(tr_mat=rat_emb,in_dim=rat_emb_dim,out_dim=feature_dim,activation=tf.nn.tanh,bias=True,kernel_reg=regularizer,name='rat_feature')
        genre_feature=dense(tr_mat=genre_emb,in_dim=genre_emb_dim,out_dim=feature_dim,activation=tf.nn.tanh,bias=True,kernel_reg=regularizer,name='genre_feature')

        concat=tf.concat([rat_emb,genre_emb],axis=1)
        rat_gate=dense(tr_mat=concat,in_dim=(genre_emb_dim+rat_emb_dim),out_dim=feature_dim,activation=tf.nn.sigmoid,bias=True,kernel_reg=regularizer,name='rat_gate')
        genre_gate=dense(tr_mat=concat,in_dim=(genre_emb_dim+rat_emb_dim),out_dim=feature_dim,activation=tf.nn.sigmoid,bias=True,kernel_reg=regularizer,name='genre_gate')


        feature_tst=rat_feature*rat_gate+genre_feature*genre_gate
        if drop_out>0:
            feature_tr=tf.nn.dropout(feature_tst,drop_out)
        else:
            feature_tr=feature_tst

        if mode.startswith('train'):
            feature=feature_tr
        else:
            feature=feature_tst
        regularizer = tf.contrib.layers.l2_regularizer(scale=lamb)

        recover=dense(tr_mat=feature,in_dim=feature_dim,out_dim=recover_dim,activation=tf.nn.relu,bias=True,kernel_reg=regularizer,name='recover')

        h1 = tf.nn.relu(recover) + 0.5
        c = tf.constant(value=5.0)
        h = tf.minimum(h1, c)

    if mode.startswith("train"):
        with tf.name_scope("train"):
            initial_learning_rate = params['lr']
            global_step = tf.Variable(epoch, trainable=False)
            if lr_decay:
                learning_rate = tf.train.exponential_decay(initial_learning_rate,
                                                       global_step=global_step,
                                                       decay_steps=10, decay_rate=0.6)
            else:
                learning_rate = initial_learning_rate

            add_global = global_step.assign_add(1)
            with tf.variable_scope("AE_TR_METRIC"):
                tr_metric = {
                    "RMSE": tf.metrics.root_mean_squared_error(tr_mat, h, weights=tr_mask)}
                tr_metric_op = tf.tuple([op for _, op in tr_metric.values()])
                # mtc=tf.summary.scalar('RMSE',tr_metric['RMSE'][1])
            loss=tf.losses.mean_squared_error(tr_mat,h,weights=tr_mask)+tf.losses.get_regularization_loss()
            opt = tf.train.AdamOptimizer(learning_rate).minimize(loss)


            # ls=tf.summary.scalar('loss',loss)

        # tr_summarys = tf.get_collection(tf.GraphKeys.SUMMARIES, scope='train')
        # tr_summary_op = tf.summary.merge([ls,mtc])


        return tr_metric_op,opt,learning_rate,add_global
    else:
        with tf.name_scope("test"):
            assert tst_mat != None
            tst_mask=tf.sign(tst_mat)
            with tf.variable_scope('AE_TST_METRIC'):
                tst_metric = {
                    "RMSE": tf.metrics.root_mean_squared_error(tst_mat, h, weights=tst_mask)}
                # mtc=tf.summary.scalar('RMSE_TST', tst_metric["RMSE"][1])
                tr_metric_op = tf.tuple([op for _, op in tst_metric.values()])

        # tst_summary_op=tf.summary.merge([mtc])
        return tr_metric_op





if __name__ == '__main__':
    if len(sys.argv)==1:
        restore_ep=0
    else:
        restore_ep=int(sys.argv[1])

    lamb=1e-4
    lr=1e-4
    rat=512
    genre=256
    feat=600
    last_layer_rglz=True

    drop_out=-1
    model_num=2

    model_params = {'lamb': lamb, 'lr': lr,'rat_emb_dim':rat,'genre_emb_dim':genre,'feature_dim':feat,'last_layer_rglz':last_layer_rglz,'drop_out':drop_out}
    summary_dir='summary/model%d/'%model_num+construct_hp_str(model_params)
    model_path='model/model%d/'%model_num+construct_hp_str(model_params)
    result_path='result/model%d/'%model_num+construct_hp_str(model_params)
    # if not os.path.exists(summary_dir):
    #     os.makedirs(summary_dir)
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    run_genre(FLAGS.train_file,FLAGS.np_file,model_params=model_params,mode="train_genre",batch_size=800,repeat_times=400,test_ep=30,restore_ep=restore_ep,save_ep=50,model_path=model_path,result_path=result_path,gpu_indx=gpu_indx)
