from .DNN_v3 import *

def NNcomponent(cfg_nn, input_layer):
    # because yaml load cannot load by order, use sorted to sort it,
    # very tricky, becarefully
    sort_net_key = sorted(cfg_nn)  
    pre_com = None
    pre_layer = input_layer
    layer_list = []
    for key in sort_net_key:
        com = cfg_nn[key]        #component
        if com['type'] == 'conv':
            # note: in_channel is pre_layer last index
            initializer   = 'truncated_normal' if 'initializer' not in com else com['initializer']
            kernel_size   = 3  if 'kernel_size' not in com else com['kernel_size']
            stride        = 2  if 'stride' not in com else com['stride']
            out_channel   = 32 if 'out_channel' not in com else com['out_channel']
            stride_ary = [1, stride, stride, 1]
            
            out_layer = Conv2D(pre_layer, kernel_size, out_channel, strides = stride_ary, name_prefix= key, initializer=initializer)
        elif  com['type'] == 'fc':
            size          = 100    if 'size' not in com else com['size']
            op            = 'relu' if 'op' not in com else com['op']
            initializer   = 'truncated_normal' if 'initializer' not in com else com['initializer']
            bias_const    = 0.01 if 'bias_const' not in com else com['bias_const']

            # print('build "{}" layer-> size={}, op={}, initializer={}, bias_const={}, pre_layer={}'.format(key, size, op, initializer, bias_const, pre_layer) )
            out_layer = FC(pre_layer, size, name_prefix = key , op=op, initializer = initializer, bias_const=bias_const)
        elif  com['type'] == 'flatten':
            out_layer = Flaten (pre_layer)
        elif  com['type'] == 'maxpool':
            out_layer = MaxPool2D(pre_layer)
        elif  com['type'] == 'dropout':
            out_layer = tf.nn.dropout(pre_layer, com['keep_prob'])

        pre_layer = out_layer
        pre_com = com
        layer_list.append(pre_layer)
    # show all layer
    # for l in layer_list:
    #     print(l)
    
    return out_layer