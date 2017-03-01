import tensorflow as tf
import numpy as np
import os
import time

from layers import *

flags   = tf.app.flags
FLAGS   = flags.FLAGS
logging = tf.logging

flags.DEFINE_float('gpu_mem', 1.0, "Fraction of gpu memory to be used.")
flags.DEFINE_float('reg', 0.001, "weight on the regularizer")
flags.DEFINE_string('name', 'unamed_run', "top level directory which contains summaries, saved models.")
flags.DEFINE_string('base_path', '/data4/girish.varma/rnncomp/', "top level directory which contains summaries, saved models.")
flags.DEFINE_integer("B", 100, "batch size")


allow_soft_placement    = True
log_device_placement    = False


def train(ctrl, model):

    step = 0
    try:
        start = time.time()
        while not ctrl['coord'].should_stop():
            step += FLAGS.B

            summ = model.train(ctrl['sess'])

            ctrl['writer'].add_summary(summ, step)
            if step % 10000 == 0:
                # print 'Epoch ', step /1000
                summ = model.test(ctrl['sess'])
                ctrl['writer'].add_summary(summ, step)
                ctrl['writer'].flush()
                ctrl['writer'].flush()

                ctrl['saver'].save(ctrl['sess'], FLAGS.base_path + "model/" + FLAGS.name, global_step = step)
                # print y_undec_[:,0]
                # print y_gt_[:,0]
                end     = time.time() - start
                print 'time for 10000 images ', end
                start   = time.time()

    except tf.errors.OutOfRangeError:

        print 'Training done'
        ctrl['saver'].save(ctrl['sess'], FLAGS.base_path + 'model/' + FLAGS.name, global_step = step)

    finally:
        ctrl['coord'].request_stop()
        ctrl['saver'].save(ctrl['sess'], FLAGS.base_path + "model/"  + FLAGS.name, global_step = step)


    ctrl['coord'].join(ctrl['threads'])
    ctrl['sess'].close()

def init():
    if not os.path.exists(FLAGS.base_path + 'model/' + FLAGS.name):
        os.makedirs(FLAGS.base_path + 'model/' + FLAGS.name)
    if not os.path.exists(FLAGS.base_path + 'summary'):
        os.makedirs(FLAGS.base_path + 'summary/' + FLAGS.name)

def create_session(writer = False, saver = False, coord = False):
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,log_device_placement=log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction=FLAGS.gpu_mem # don't hog all vRAM
    config.gpu_options.allow_growth=True
    output = {}
    sess = tf.Session(config=config)
    output['sess'] = sess
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    if writer:
        output['writer'] = tf.summary.FileWriter(FLAGS.base_path +  'summary/' + FLAGS.name, sess.graph)
    if saver:
        output['saver'] = tf.train.Saver()
    if coord:
        output['coord'] = coord = tf.train.Coordinator()
        output['threads'] = tf.train.start_queue_runners(sess=sess, coord=coord)
    return output


def match(y, y_pred, name = 'match'):
    with tf.variable_scope(name):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1)), tf.float32))
        return accuracy


def softmax_cross_entropy(y, y_pred, name = 'softmax_cross_entropy'):
    return tf.losses.softmax_cross_entropy(y, y_pred)

losses = {
    'softmax_cross_entropy': softmax_cross_entropy
}

accuracies = {
    'class_match' : match
}

def classify(y, y_pred, y_test = None, y_pred_test = None, **kwargs):
    loss = kwargs.get('loss', 'softmax_cross_entropy')
    acc  = kwargs.get('acc', 'class_match')
    with tf.variable_scope('train_loss_acc'):
        train_loss = losses[loss](y, y_pred)
        train_acc  = accuracies[acc](y, y_pred)
        train_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_train_loss = train_loss + FLAGS.reg*train_reg
    
        train_summary = [
            tf.summary.scalar('loss', train_loss),
            tf.summary.scalar('accuracy', train_acc),
            tf.summary.scalar('regularizer', train_reg),
            tf.summary.scalar('total_loss', total_train_loss)
        ]


    if y_test != None:
        with tf.variable_scope('test_loss_acc'):
            test_loss = losses[loss](y_test, y_pred_test)
            test_acc  = accuracies[acc](y_test, y_pred_test)
            test_summary = [
                tf.summary.scalar('loss', test_loss),
                tf.summary.scalar('accuracy', test_acc)
            ]

    optimizer = kwargs.get('optimizer', 'adam')
    optimizer = minimize(total_train_loss, rate = kwargs.get('rate', 0.001), algo = optimizer)
    return optimizer, train_summary, test_summary


def minimize(loss, **kwargs):
    algo = kwargs.get('algo', 'adam')
    rate = kwargs.get('rate', 0.001)
    name = kwargs.get('name', 'optimizer')
    with tf.variable_scope(name):
        if algo == 'adam':
            return tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
        elif algo == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss)
        elif algo == 'momentum':
            return tf.train.MomentumOptimizer(rate, kwargs.get('momentum', None)).minimize(loss)

def sequential(x, net, defaults = {}, name = '', reuse = None, var = {}, layers = {}):
    layers = dict(layers.items() + predefined_layers.items())
    y = x
    logging.info('Building Sequential Network : %s', name)
    
    with tf.variable_scope(name, reuse = reuse):
        for i in range(len(net)):
            ltype   = net[i][0]
            lcfg    = net[i][1] if len(net[i]) == 2 else {}
            lname   = lcfg.get('name', ltype + str(i))
            ldefs   = defaults.get(ltype, {})
            lcfg    = dict(ldefs.items() + lcfg.items())
            for k, v in lcfg.iteritems():
                if isinstance(v, basestring) and v[0] == '$':
                    # print var, v
                    lcfg[k] = var[v[1:]]
            y  = layers[ltype](y, lname, **lcfg)
            logging.info('\t %s \t %s', lname, y.get_shape().as_list())
        return y

