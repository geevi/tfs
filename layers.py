import tensorflow as tf

def batch_norm(x, name = 'batch_norm', **kwargs):
    momentum = kwargs.get('batch_norm_momentum', 0.9)
    training = kwargs.get('training', True)
    axis = kwargs.get('axis', -1)
    # y = tf.contrib.layers.batch_norm(x, is_training=training, scale=True, updates_collections=None, decay=momentum, name)
    y = tf.layers.batch_normalization(x, axis = axis, name = name, training = training, momentum = momentum)
    return y

def activation(x, name, **kwargs):
    atype = kwargs.get('func', 'relu')
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
        y = batch_norm(y, lname + '_batch_norm', axis = 3, **lcfg)
    if act:
        y = activation(y, name = lname + '_' + act, type = act, alpha = lcfg.get('alpha', None))

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
    return tf.layers.dropout(x, pr, name = name)

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
    'normalize' : normalize,
    'batch_norm': batch_norm,
    'activation': activation
}
