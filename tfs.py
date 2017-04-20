from __future__ import print_function
from __future__ import absolute_import
from builtins import next
from builtins import str
from builtins import range
from past.builtins import basestring

import tensorflow as tf
import numpy as np
import os
import shutil
import time

from .layers import *
from .classes import *

flags   = tf.app.flags
FLAGS   = flags.FLAGS
logging = tf.logging

flags.DEFINE_float('gpu_mem', 1.0, "Fraction of gpu memory to be used.")
flags.DEFINE_float('reg', 1e-8, "weight on the regularizer")
flags.DEFINE_string('name', 'unamed_run', "top level directory which contains summaries, saved models.")
flags.DEFINE_string('base_path', '/home/girish.varma/yt8m/', "top level directory which contains summaries, saved models.")
flags.DEFINE_integer("B", 30, "batch size")
flags.DEFINE_float("rate", 0.001, "learning rate")
flags.DEFINE_bool("new", False, "Delete the previous run with same name and start new.")


allow_soft_placement    = True
log_device_placement    = False

def training_loop(ctrl, model, test = False):

    step = 0
    try:
        start = time.time()
        while not ctrl['coord'].should_stop():
            try:
                summ, step = model.train(ctrl['sess'])
                ctrl['writer'].add_summary(summ, step*FLAGS.B)
                if step % 10 == 0:
                     ctrl['writer'].flush()
                     ctrl['writer'].flush()
                if step % 10 == 0 and test:
                    summ = model.validate(ctrl['sess'])
                    ctrl['writer'].add_summary(summ, step*FLAGS.B)
                    ctrl['writer'].flush()
                    ctrl['writer'].flush()
                    ctrl['saver'].save(ctrl['sess'], FLAGS.base_path + "model/" + FLAGS.name + '/', global_step = step)
                    end = time.time() - start
                    #print 'time for 10 steps ', end, '. Samples seen ', step *FLAGS.B
                    start   = time.time()

            except tf.errors.DataLossError as err:
                print(err.message)


    except tf.errors.OutOfRangeError:

        print('Training done')
        ctrl['saver'].save(ctrl['sess'], FLAGS.base_path + 'model/' + FLAGS.name, global_step = step)

    finally:
        ctrl['coord'].request_stop()
        ctrl['saver'].save(ctrl['sess'], FLAGS.base_path + "model/"  + FLAGS.name, global_step = step)


    ctrl['coord'].join(ctrl['threads'])
    ctrl['sess'].close()


def find_class_by_name(name, modules):
    """Searches the provided modules for the named class and returns it."""
    modules = [getattr(module, name, None) for module in modules]
    return next(a for a in modules if a)

def init():
    if not os.path.exists(FLAGS.base_path + 'model'):
        os.makedirs(FLAGS.base_path + 'model')
    if not os.path.exists(FLAGS.base_path + 'summary'):
        os.makedirs(FLAGS.base_path + 'summary')

    model_path = FLAGS.base_path + 'model/' + FLAGS.name
    summary_path = FLAGS.base_path + 'summary/' + FLAGS.name

    if os.path.exists(model_path) and not FLAGS.load:
        shutil.rmtree(model_path)

    if os.path.exists(summary_path) and not FLAGS.load:
        shutil.rmtree(summary_path)

    if not os.path.exists(model_path):
        os.makedirs(model_path)
    if not os.path.exists(summary_path):
        os.makedirs(summary_path)


def session():
    config = tf.ConfigProto(allow_soft_placement=allow_soft_placement,log_device_placement=log_device_placement)
    config.gpu_options.per_process_gpu_memory_fraction=FLAGS.gpu_mem # don't hog all vRAM
    config.gpu_options.allow_growth=True
    
    sess = tf.Session(config=config)
    return sess


def create_session(writer = False, saver = False, coord = False):
    sess = session()
    output = {'sess': sess}
    init = tf.global_variables_initializer()
    sess.run(init)
    sess.run(tf.local_variables_initializer())
    if writer:
        output['writer'] = tf.summary.FileWriter(FLAGS.base_path +  'summary/' + FLAGS.name, sess.graph)
    if saver:
        output['saver'] = tf.train.Saver(var_list= tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES))
    if coord:
        output['coord'] = coord = tf.train.Coordinator()
        output['threads'] = tf.train.start_queue_runners(sess=sess, coord=coord)
    return output

def shape(x):
    return x.get_shape().as_list()

def match(y, y_pred, name = 'match'):
    with tf.variable_scope(name):
        accuracy = tf.reduce_mean(tf.cast(tf.equal(tf.argmax(y_pred,1), tf.argmax(y,1)), tf.float32))
        return accuracy


def softmax_cross_entropy(y, y_pred, name = 'softmax_cross_entropy'):
    return tf.losses.softmax_cross_entropy(y, y_pred)


def classify(y, y_pred, y_valid = None, y_pred_valid = None, **kwargs):
    loss = kwargs.get('loss', softmax_cross_entropy)
    acc = kwargs.get('acc', match)
    
    with tf.variable_scope('train_loss_acc'):
        train_loss = loss(y, y_pred)
        train_acc  = acc(y, y_pred)
        train_reg = sum(tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES))
        total_train_loss = train_loss + FLAGS.reg*train_reg
    
        train_summary = [
            tf.summary.scalar('loss', train_loss),
            tf.summary.scalar('accuracy', train_acc),
            tf.summary.scalar('regularizer', train_reg),
            tf.summary.scalar('total_loss', total_train_loss)
        ]

    optimizer, rate, global_step = minimize(total_train_loss, **kwargs)
    train_summary += [tf.summary.scalar('learning_rate', rate)]

    valid_summary = []
    if y_valid != None:
        with tf.variable_scope('valid_loss_acc'):
            valid_loss = loss(y_valid, y_pred_valid)
            valid_acc  = acc(y_valid, y_pred_valid)
            valid_summary += [
                tf.summary.scalar('loss', valid_loss),
                tf.summary.scalar('accuracy', valid_acc)
            ]

    
    return optimizer, train_summary, valid_summary, global_step


def minimize(loss_tensor, **kwargs):
    algo = kwargs.get('algo', 'adam')
    rate = kwargs.get('rate', 0.01)
    name = kwargs.get('name', 'optimizer')
    grad_clip = kwargs.get('grad_clip', 1.0)
    global_step = tf.Variable(0, trainable=False, name="global_step")
    learning_rate = tf.train.exponential_decay(
      rate,
      global_step * FLAGS.B,
      kwargs.get('learning_rate_decay_examples', 4000000),
      kwargs.get('learning_rate_decay', 0.95),
      staircase=True)
    with tf.variable_scope(name):
        optimizer = None
        if algo == 'adam':
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        elif algo == 'sgd':
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
        elif algo == 'momentum':
            optimizer = tf.train.MomentumOptimizer(learning_rate, kwargs.get('momentum', None))

        gvs = optimizer.compute_gradients(loss_tensor)
        if grad_clip != 0. :
            gvs = [(tf.clip_by_value(grad, -grad_clip, grad_clip), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(gvs, global_step = global_step)
        return train_op, learning_rate, global_step

def sequential(x, net, defaults = {}, name = '', reuse = None, var = {}, layers = {}):
    layers = dict(list(layers.items()) + list(predefined_layers.items()))
    y = x
    logging.info('Building Sequential Network : %s', name)
    
    with tf.variable_scope(name, reuse = reuse):
        for i in range(len(net)):
            ltype   = net[i][0]
            lcfg    = net[i][1] if len(net[i]) == 2 else {}
            lname   = lcfg.get('name', ltype + str(i))
            ldefs   = defaults.get(ltype, {})
            lcfg    = dict(list(ldefs.items()) + list(lcfg.items()))
            for k, v in list(lcfg.items()):
                if isinstance(v, basestring) and v[0] == '$':
                    # print var, v
                    lcfg[k] = var[v[1:]]
            y  = layers[ltype](y, lname, **lcfg)
            logging.info('\t %s \t %s', lname, y.get_shape().as_list())
        return y

