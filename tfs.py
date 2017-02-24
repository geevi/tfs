import tensorflow 
import numpy 
import os
from tensorflow.examples.tutorials.mnist import input_data
import time

time    = time
tf      = tensorflow
np      = numpy
flags   = tf.app.flags
FLAGS   = flags.FLAGS
logging = tf.logging

flags.DEFINE_float('gpu_mem', 1.0, "Fraction of gpu memory to be used.")
flags.DEFINE_string('summ_dir', 'summary', "directory for putting the summaries.")
flags.DEFINE_string('name', 'unamed_run', "top level directory which contains summaries, saved models.")
flags.DEFINE_string("working_directory", "", "")
flags.DEFINE_integer("B", 100, "batch size")



allow_soft_placement    = True
log_device_placement    = False


def mnist():
    data_directory = os.path.join(FLAGS.working_directory, "MNIST")
    if not os.path.exists(data_directory):
        os.makedirs(data_directory)
    mnist = input_data.read_data_sets(data_directory, one_hot=True)
    return mnist



def init():
    if not os.path.exists('saved/' + FLAGS.name):
        os.makedirs('saved/' + FLAGS.name)
    if not os.path.exists('summary'):
        os.makedirs('summary')

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
        output['writer_train'] = tf.summary.FileWriter(FLAGS.summ_dir + '/' + FLAGS.name + '_train', sess.graph)
        output['writer_test'] = tf.summary.FileWriter(FLAGS.summ_dir + '/' + FLAGS.name + '_test')
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

def classify(y, y_pred, **kwargs):
    loss = kwargs.get('loss', 'softmax_cross_entropy')
    loss = losses[loss](y, y_pred)
    tf.summary.scalar('loss', loss)
    acc  = kwargs.get('acc', 'class_match')
    acc  = accuracies[acc](y, y_pred)
    tf.summary.scalar('accuracy', acc)
    optimizer = kwargs.get('optimizer', 'adam')
    optimizer = minimize(loss, rate = kwargs.get('rate', 0.001), algo = optimizer)
    summary = tf.summary.merge_all()
    return optimizer, summary


def minimize(loss, **kwargs):
    algo = kwargs.get('algo', 'adam')
    rate = kwargs.get('rate', 0.001)
    name = kwargs.get('name', 'train')
    with tf.variable_scope(name):
        if algo == 'adam':
            return tf.train.AdamOptimizer(learning_rate=rate).minimize(loss)
        elif algo == 'sgd':
            return tf.train.GradientDescentOptimizer(learning_rate=rate).minimize(loss)
        elif algo == 'momentum':
            return tf.train.MomentumOptimizer(rate, kwargs.get('momentum', None)).minimize(loss)


def _convs(y, lname, ltype, lcfg):
    chan    = lcfg.get('chan', y.get_shape().as_list()[-1])
    window  = lcfg.get('window', 3)
    bias    = lcfg.get('bias', True)
    act     = lcfg.get('act', None)
    bn      = lcfg.get('batch_norm', False)
    stride  = lcfg.get('stride', 1)
    padding = lcfg.get('padding', 'same')
    
    if ltype == 'deconv':
        y = tf.layers.conv2d_transpose(y, chan, window, strides= stride, padding = padding, use_bias = bias, name = lname)
    elif ltype == 'conv':
        y = tf.layers.conv2d(y, chan, window, strides= stride, padding = padding, use_bias = bias, name = lname)
    if bn:
        momentum = lcfg.get('batch_norm_momentum', 0.9)
        training = lcfg.get('training', True)
        y = tf.contrib.layers.batch_norm(y, is_training=training, scale=True, updates_collections=None, decay=momentum)
        # y = tf.layers.batch_normalization(y , axis = 3, name = lname + '_batch_norm', training = training, momentum = momentum)

    if act:
        y = activation(y, name = lname + '_' + act, type = act, alpha = lcfg.get('alpha', None))

    return y



def activation(x, name, **kwargs):
    atype = kwargs.get('type', 'relu')
    if atype == 'relu':
        y = tf.nn.relu(x, name = name)
    elif atype == 'tanh':
        y = tf.nn.tanh(x, name = name )
    elif atype == 'leaky_relu':
        alpha = kwargs.get('alpha', 0.)
        y = tf.maximum(alpha*x, x, name = name)
    elif atype == 'sigmoid':
        y = tf.nn.sigmoid( x, name = name )
    return y

def reshape(x, name, **kwargs):
    shape    = kwargs.get('shape', kwargs.get('shape'))
    if shape == None:
        shape = [-1, reduce(lambda a,b: a*b, x.get_shape().as_list()[1:])]
    return tf.reshape(x, shape, name = name)

def deconv(x, name, **kwargs):
    return _convs(x, name, 'dconv', kwargs)

def conv(x, name, **kwargs):
    return _convs(x, name, 'conv', kwargs)

def dropout(x, name, **kwargs):
    pr = kwargs.get('pr', 0.5)
    return tf.layers.dropout(y, pr, name = name)

def pool(x, name, **kwargs):
    window  = kwargs.get('window', 2)
    stride  = kwargs.get('stride', 2)
    padding = kwargs.get('padding', 'same')
    return tf.layers.max_pooling2d(x, window, stride, padding, name = name)

def dense(x, name, **kwargs):
    units = kwargs.get('units', None)
    act = kwargs.get('act', None)
    y =  tf.layers.dense(x, units, name = name)
    if act:
        y = activation(y, name+ '_' + act, type =act, alpha = kwargs.get('alpha', None))
    return y

def normalize(x, name = 'normalize', **kwargs):
    mean = kwargs.get('mean', 0)
    variance = kwargs.get('variance', 1)
    with tf.name_scope(name):
        y = (x - mean) / variance
        return y


predefined_layers = {
    'reshape'   : reshape,
    'conv'      : conv,
    'deconv'    : deconv,
    'dropout'   : dropout,
    'pool'      : pool,
    'dense'     : dense,
    'normalize' : normalize
}

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

