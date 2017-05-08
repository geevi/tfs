from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from builtins import range
from functools import reduce
from tensorflow.contrib.layers.python.layers import initializers

import tensorflow as tf
from .tfs import *

def l2(W):
    return tf.reduce_sum(tf.square(W))

def kernel_init():
    
    return initializers.xavier_initializer()

def bias_init():
    return tf.constant_initializer(0.0)


def lstm_layer(x, name = 'lstm', **kwargs):
    with tf.variable_scope(name, reuse=kwargs.get('reuse', False)):
        hidden_size = kwargs.get('units', 500)
        ret = kwargs.get('ret', 'outputs')
        seq_lens = kwargs.get('seq_lens')
        
        cell = tf.contrib.rnn.BasicLSTMCell(hidden_size, forget_bias=1.0, state_is_tuple=False)

        outputs, state = tf.nn.dynamic_rnn(cell, x, sequence_length=seq_lens, dtype=tf.float32, time_major = True)

        if ret == 'outputs':
            return outputs
        else:
            return state


def seq_linear_layer(x, name, **kwargs):
    seq_len,_,size = x.get_shape().as_list()
    units = kwargs.get('units')
    
    with tf.variable_scope(name):
        W = tf.get_variable('weights', [size, units])
        b = tf.get_variable('bias', [units])
        a = []
        for i in range(seq_len):
            a.append(tf.matmul(x[i],W) + b)
            # print 'a[i]', shape(a[i])
        a= tf.stack(a, axis = 0) #time x batch_size 
        return a

def seq_softmax(x, name = 'seq_sotmax', **kwargs):
    # x has shape time x batch x 1
    with tf.variable_scope(name):
        seq_lens = kwargs.get('seq_lens')
        max_frames = x.get_shape().as_list()[0]
        mask = tf.cast(tf.sequence_mask(seq_lens, max_frames), tf.float32)# batch size x max time
        x = tf.transpose(x, [1, 0, 2])
        a = tf.reduce_sum(x, axis = 2)
        # print a.get_shape().as_list()
        a = tf.nn.softmax(a)
        a = a * mask
        a = a / tf.reduce_sum(a, axis = 1, keep_dims = True)
        return a




def batch_norm(x, name = 'batch_norm', **kwargs):
    momentum = kwargs.get('batch_norm_momentum', 0.9)
    training = kwargs.get('training', True)
    axis = kwargs.get('axis', -1)
    y = tf.contrib.layers.batch_norm(x, is_training=training, scale=True, updates_collections=None, decay=momentum)
    # y = tf.layers.batch_normalization(x, axis = axis, name = name, training = training, momentum = momentum)
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
        #print 'sigmoid'
        y = tf.sigmoid( x, name = name )
    return y

def _convs(y, ltype, lname = '', **lcfg):
    chan    = lcfg.get('chan', y.get_shape().as_list()[-1])
    window  = lcfg.get('window', 3)
    bias    = lcfg.get('bias', True)
    act     = lcfg.get('act', None)
    bn      = lcfg.get('batch_norm', False)
    stride  = lcfg.get('stride', 1)
    padding = lcfg.get('padding', 'same')
    
    with tf.variable_scope(lname):
        if ltype == 'deconv':
            y = tf.layers.conv2d_transpose(y, chan, window, strides= stride, padding = padding, use_bias = bias,  
            kernel_initializer = kernel_init(), bias_initializer = bias_init(), kernel_regularizer = l2)
        elif ltype == 'conv':
            y = tf.layers.conv2d(y, chan, window, strides= stride, padding = padding, use_bias = bias, 
            kernel_initializer = kernel_init(), bias_initializer = bias_init(), kernel_regularizer = l2)
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

def deconv(x, name = '', **kwargs):
    return _convs(x, 'deconv', lname = name, **kwargs)

def conv(x, name = '', **kwargs):
    return _convs(x, 'conv', lname = name, **kwargs)

def dropout(x, name = '', **kwargs):
    pr = kwargs.get('pr', 0.5)
    return tf.layers.dropout(x, pr, name = name)

def pool(x, name = '', **kwargs):
    window  = kwargs.get('window', 2)
    stride  = kwargs.get('stride', 2)
    padding = kwargs.get('padding', 'same')
    return tf.layers.max_pooling2d(x, window, stride, padding, name = name)

def dense(x, name='', **kwargs):
    with tf.variable_scope(name, reuse = kwargs.get('reuse', False)):
        units = kwargs.get('units', None)
        act = kwargs.get('act', None)
        y =  tf.layers.dense(x, units, name = name, 
        kernel_initializer = kernel_init(), bias_initializer = bias_init(), kernel_regularizer = l2)
        if act:
            y = activation(y, name+ '_' + act, func =act, alpha = kwargs.get('alpha', None))
        return y

def normalize(x, name = 'normalize', **kwargs):
    mean = kwargs.get('mean', 0)
    variance = kwargs.get('variance', 1)
    with tf.name_scope(name):
        y = (x - mean) / variance
        return y

def sampleWhiten(x, name = 'sampleWhiten', **kwargs):
    mVal,stdVal = tf.nn.moments(x, axes=(1,2,3), keep_dims=True)
    with tf.name_scope(name):
        y= (x-mVal) / (stdVal+0.00001)
        return y

def flatten(x, name = 'flatten', **kwargs):
    return reshape(x, name, shape = None)

predefined_layers = {
    'reshape'   : reshape,
    'conv'      : conv,
    'deconv'    : deconv,
    'dropout'   : dropout,
    'pool'      : pool,
    'dense'     : dense,
    'normalize' : normalize,
    'sampleWhiten' : sampleWhiten,
    'batch_norm': batch_norm,
    'activation': activation,
    'lstm'      : lstm_layer,
    'seq_linear': seq_linear_layer,
    'seq_softmax': seq_softmax,
    'flatten'   : flatten
}
