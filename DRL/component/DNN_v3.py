import tensorflow as tf
import numpy as np 

def safe_get(name, *args, **kwargs):
    """ Same as tf.get_variable, except flips on reuse_variables automatically """
    try:
        # print('New variable -> {} '.format(name))
        return tf.get_variable(name, *args, **kwargs)
    except ValueError as e :
        # print('ValueError as e  -> ' + str(e))
        print('REuse variable -> {}, scope={} '.format(name, str(tf.get_variable_scope())  ))
        tf.get_variable_scope().reuse_variables()
        return tf.get_variable(name, *args, **kwargs)


def weight_variable(shape, name="w", initializer = 'xavier', normal_mean=0.0, normal_stddev=0.01):
    # print('name ={} weight_variable shape = {}'.format(name, shape))
    if initializer == 'xavier':
        xavier_init =  tf.contrib.layers.xavier_initializer(dtype=tf.float32)
        return safe_get(name, shape, initializer=xavier_init, dtype=tf.float32)
    elif initializer == 'truncated_normal':
        truncated_normal_init = tf.truncated_normal_initializer(mean=normal_mean, stddev=normal_stddev, dtype=tf.float32)
        return safe_get(name, shape, initializer=truncated_normal_init, dtype=tf.float32)
    elif initializer == 'selu':
        muti_exclude_last = np.prod(shape[:-1])  # ex [4, 4, 3, 32] -> 4 * 4*3
        weights = np.random.normal(scale=np.sqrt(1.0/muti_exclude_last), size=shape).astype('f')
        return safe_get(name, list(shape), initializer=tf.constant_initializer(weights), dtype=tf.float32)
    else:
        assert False, 'weight_variable() say Error initializer'

def bias_variable(shape, name = "b", const = 0.0):
    if const ==0:
        init = tf.zeros(shape, dtype=tf.float32)
    else:
        init = tf.constant(const, shape=shape, dtype=tf.float32)
    return safe_get(name, initializer=init)
    # return safe_get(name, initializer=tf.zeros(shape, dtype=tf.float32))
    # inital = tf.constant(0.1, shape=shape)
    # return tf.Variable(inital, name = name)

def Conv2D(x , kernel_size = 3, out_channel = 32, in_channel = None, name_prefix = "conv", 
            w = None, b = None, initializer= 'xavier',strides = [1,2,2,1]):
    if in_channel is None:
        assert len(x.shape) == 4, 'Conv2D() say the len of input shape is not 4 %s' % name_prefix
        in_channel = int(x.shape[3])

    # w and b
    if w==None:
        w = weight_variable([kernel_size, kernel_size, in_channel, out_channel] , 
                            name= name_prefix + "_w", initializer=initializer) 
    if b==None:
        b = bias_variable([out_channel]  , name= name_prefix + "_b")

    #Combine
    #return tf.nn.relu(tf.nn.conv2d(x, w, strides=strides, padding='SAME')+ b, name = name_prefix) #output size 28x28x32
    layer = tf.nn.conv2d(x, w, strides=strides, padding='SAME')+ b
    # try:
    #     layer = tf.nn.relu(tf.contrib.layers.layer_norm(layer, center=True,
    #         scale=False)) # updates_collections=None
    # except ValueError:
    #     print('in layer_norm Value Error')
    #     layer = tf.nn.relu(tf.contrib.layers.layer_norm(layer, center=True,
    #         scale=False, reuse=True))
    
    # return layer
    return norm(layer = layer, norm_type='layer_norm', name = name_prefix)

def MaxPool2D(x, pool_size = 2):
    return tf.nn.max_pool(x, ksize=[1, pool_size, pool_size, 1],
                        strides=[1, pool_size, pool_size, 1], padding='SAME')

def Flaten(x):
    assert len(x.shape) == 4, 'flat() say the len of input shape is not 4'
    num = int(x.shape[1]) * int(x.shape[2]) * int(x.shape[3])
    # print('flat num = %d' % num)
    return tf.reshape(x, [-1, num]) 

def FC(x, fc_size = 1024, name_prefix = 'fc', w = None, b = None, initializer='xavier', op='relu', bias_const=0):
    assert len(x.shape) == 2, 'FC() say the len of input shape is not 2'
    num = int(x.shape[1]) 

    if w == None:
        w = weight_variable([num, fc_size],name= name_prefix + "_w", initializer=initializer) 
    if b == None:
        b = bias_variable([fc_size], name = name_prefix + '_b', const=bias_const) 
    
    if op=='relu':
        return tf.nn.relu(tf.matmul(x, w) + b, name = name_prefix+ "_relu")
    elif op=='softmax':
        return tf.nn.softmax(tf.matmul(x, w) + b, name = name_prefix+ "_softmax")
    elif op=='tanh':
        return tf.nn.tanh(tf.matmul(x, w) + b, name = name_prefix+ "_tanh")
    elif op=='none':
        return tf.matmul(x, w) + b
    else:
        print('error op ==' + op)
        assert False, 'FC() say Error op'
   

def norm(layer, norm_type='batch_norm', decay=0.9, id=0, is_training=True, activation_fn=tf.nn.relu, name='conv_'):
    # from https://github.com/tianheyu927/mil/blob/master/tf_utils.py
    if norm_type != 'batch_norm' and norm_type != 'layer_norm':
        return tf.nn.relu(layer)
    with tf.variable_scope('norm_layer_%s' % (name)) as vs:
        if norm_type == 'batch_norm':
            if is_training:
                try:
                    layer = tf.contrib.layers.batch_norm(layer, is_training=True, center=True,
                        scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs) # updates_collections=None
                except ValueError:
                    layer = tf.contrib.layers.batch_norm(layer, is_training=True, center=True,
                        scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs, reuse=True) # updates_collections=None
            else:
                layer = tf.contrib.layers.batch_norm(layer, is_training=False, center=True,
                    scale=False, decay=decay, activation_fn=activation_fn, updates_collections=None, scope=vs, reuse=True) # updates_collections=None
        elif norm_type == 'layer_norm': # layer_norm
            # Take activation_fn out to apply lrelu
            try:
                layer = activation_fn(tf.contrib.layers.layer_norm(layer, center=True,
                    scale=False, scope=vs)) # updates_collections=None
                
            except ValueError as e: 
                print('ValueError ->', e)
                print('in layer_norm Value Error')
                layer = activation_fn(tf.contrib.layers.layer_norm(layer, center=True,
                    scale=False, scope=vs, reuse=True))
        # elif norm_type == 'selu':
        #     layer = selu(layer)
        else:
            raise NotImplementedError('Other types of norm not implemented.')
        return layer