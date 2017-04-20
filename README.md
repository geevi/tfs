# tfs
A small, simple library to make tensorflow easy.

## Usage
git clone https://github.com/geevi/tfs.git

Then add the following import line 


```python

from tfs import *

```

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
    
def my_seq_softmax(x, name, **kwargs):
    time_steps = kwargs.get('time_steps')
    y = []
    for i in range(time_steps):
        y.append(tf.nn.softmax(x[i]))

    y = tf.stack(y, axis=0)
    return y

    
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


net = [
        ['dense', {
            'units'         : 2000
        }],
        ['dense', {
            'units'         : 2,
            'act'           : None #default parameter overriden
        }]
]

string = sequential(class_one_hot, net, defaults = defaults, name = 'class2str')
```

- All basic layers implemented with proper variable scoping so that tensorboard graph view looks nice.

- Create summaries, and saved models, with paths customizable by project and base dir.

- Base classes for Datasets as well Models

```python
class Logistic(BaseModel):

    def __init__(self, dataset):

        self.dataset = dataset

        net = [
           ['dense', {
               'units'     : FLAGS.hidden,
               'act'       : 'relu'
           }],
            ['dense', {
                'units'     : dataset.num_classes,
                'act'       : 'sigmoid'
            }]
        ]

        self.logits         = sequential(dataset.mean_rgb, net, name='mlp')
        self.logits_valid   = sequential(dataset.mean_rgb_valid, net, reuse=True, name='mlp')
        self.logits_test    = sequential(dataset.mean_rgb_test, net, reuse=True, name='mlp')

        args = {
            'y'             : dataset.labels,
            'y_pred'        : self.logits,
            'y_test'        : dataset.labels_valid,
            'y_pred_test'   : self.logits_valid,
            'rate'          : FLAGS.rate,
            'loss'          : cross_entropy,
            'acc'           : hit_at_1
        }
        
        self.optimizer, train_summary, test_summary = classify(**args)
        self.train_summary_op   = tf.summary.merge(train_summary)
        self.test_summary_op    = tf.summary.merge(test_summary)


```

- Training and Testing Loops, Helper functions that create session, coordinator, file writer etc. for you.


```python
def main(_):
    init()
    dataset = yt8m.YT8M()
    model = simple.Logistic(dataset)

    ctrl = create_session(saver = True, writer = True, coord = True)

    if not FLAGS.load:
        training_loop(ctrl, model, test=True)
    else:
        testing_loop(ctrl, model, dataset)
```

## FAQ
- How is this different from Keras, Sonet, tflearn, slim etc?

    - This is a very small and easy to understand library compared to them (a very thin wrapper). 
    - It still reduces the code one needs to write for building models in tensorflow by a large extend. 
    - Adding a custom layer is simply defining a function (instead of a class). 
    - The customizable default parameters (which is a nice feature from slim) is there. 
    - The model definition in json form is much more readable compared to other formats (IMHO). 
    - Easy to mix and match with regular tensorflow code.

## Contributions are welcome
