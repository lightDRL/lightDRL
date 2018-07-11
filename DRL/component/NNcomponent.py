from DNN_v3 import *

def NNcomponent(cfg_nn, input_layer):
    # because yaml load cannot load by order, use sorted to sort it,
    # very tricky, becarefully
    sort_net_key = sorted(cfg_nn)  
    pre_com = None
    pre_layer = None
    fc_down_factor = 0
    for key in sort_net_key:
        com = cfg_nn[key]        #component
        
        if com['type'] == 'conv':
            if pre_com == None:
                in_channel = self.img_d
            elif pre_com['type'] == 'conv':
                in_channel = pre_com['out_channel']
            else:
                print('build_weight() say Error component property, in conv else ')
            w_shape = [com['kernel_size'], com['kernel_size'], in_channel, com['out_channel']]
            com['w'] = weight_variable(w_shape , name= key + "_w") 
            com['b'] = bias_variable([com['out_channel']]  , name= key + "_b")
            fc_down_factor = fc_down_factor * int(com['stride']) if fc_down_factor is not 0 else  int(com['stride'])
        elif  com['type'] == 'fc':
            if pre_com==None:
                pre_layer = input_layer
            elif pre_com['type'] == 'conv':    
                in_channel =  math.ceil(self.img_w/ fc_down_factor) * math.ceil(self.img_h/ fc_down_factor) * pre_com['out_channel'] #because default padding is 'SAME'
                print('first layer')
                print('\t fc_down_factor = '+ str(fc_down_factor))
                print('\t in_channel = '+ str(in_channel))

            elif pre_com['type'] == 'fc':
                in_channel =  pre_com['size']
            else:
                print('build_weight() say Error component property, in conv else ')

            size          = 100    if 'size' not in com else com['size']
            op            = 'relu' if 'op' not in com else com['op']
            initializer   = 'truncated_normal' if 'initializer' not in com else com['initializer']
            bias_const    = 0.01 if 'bias_const' not in com else com['bias_const']

            # print('build "{}" layer-> size={}, op={}, initializer={}, bias_const={}'.format(key, size, op, initializer, bias_const) )
            out_layer = FC(pre_layer, size, name_prefix = key , op=op, initializer = initializer, bias_const=bias_const)
    

        pre_layer = out_layer
        pre_com = com

    return out_layer