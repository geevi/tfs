# tfs
A small, simple library to make tensorflow easy.


## Features

- Building Sequential Network using json representation 

```python
x = tf.placeholder(tf.float32, [1, 128,128, 1])

net = [
    ['conv', {
        'window'    : 5,
        'chan'      : 32
    }],
    ['pool'],
    ['conv', {
        'window'    : 5,
        'chan'      : 64
    }],
    ['deconv', {
        'chan'      : 64,
        'window'    : 5,
        'stride'    : 2
    }],
    ['deconv', {
        'chan'      : 32,
        'window'    : 5,
        'stride'    : 2
    }],
    ['deconv', {
        'chan'      : 10,
        'window'    : 5,
        'stride'    : 2,
        'act'       : 'sigmoid'
    }]
]

self.logits = sequential(x, net)
```

- Build custom layers by defining a function


```python

def my_lstm_layer(x, name, **kwargs):
    time_steps = kwargs.get('time_steps')
    units = kwargs.get('units')
    inputs = []
    for i in range(time_steps):
        inputs.append(x)

    inputs = tf.stack(inputs, axis=0)

    return lstm_layer(inputs, units = units, seq_lens = tf.constant(time_steps, shape=[FLAGS.B]))
    
layers = {
        'my_lstm_layer'     : my_lstm_layer,
        'my_seq_softmax'    : my_seq_softmax
    }


net = [
    ['dense', {
        'units'         : 500
    }],
    ['my_lstm_layer', {
        'units'         : 500,
        'time_steps'    : FLAGS.L
    }],
    ['seq_linear', {
        'units'         : 2,
        'len'           : FLAGS.L
    }],
    ['my_seq_softmax', {
        'time_steps'    : FLAGS.L
    }]
]

string = sequential(class_one_hot, net, defaults = defaults, layers = layers, name = 'class2str')
    
```

- Set default parameters for the same type of layers, and override if needed.


```python
defaults = {
        'dense'     : {
        'act'           : 'relu'
        }
}

layers = {
        'rep_seq_linear'    : rep_seq_linear
}

net = [
        ['dense', {
            'units'         : 2000
        }],
        ['dense', {
            'units'         : 2,
            'act'           : None #default parameter overriden
        }]
]

string = sequential(class_one_hot, net, defaults = defaults, layers = layers, name = 'class2str')
```

- All basic layers implemented with proper variable scoping so that tensorboard graph view looks nice.

``

- Create summaries, and saved models, with paths customizable by project and base dir.

- Base classes for Datasets as well Models

- Training and Testing Loops

- Helper functions that create session, coordinator, file writer etc. for you.
