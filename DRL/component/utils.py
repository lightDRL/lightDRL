import tensorflow as tf
import random
import numpy as np

def print_tf_var(head_str = 'print_tf_var', graph = None):
    print('graph = ', graph)
    graph = tf.get_default_graph() if graph == None else graph
    
    with graph.as_default(): 
        print('START=============print_all_var(), %s===============' % head_str)
        print('---------tf.trainable_variables()---------')
        train_var = tf.trainable_variables()
        for v in train_var:
            print(v)
            # print(v.name)


        print('---------tf.GLOBAL_VARIABLES()---------')
        get_collection_var = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
        for v in get_collection_var:
            print(v)

        print('---------get all Placeholder---------')
        for x in tf.get_default_graph().get_operations() :
            if  "Placeholder" in x.type:
                print('name = {}'.format( x.name ) )
                tensor = tf.get_default_graph().get_tensor_by_name(x.name+":0")
                print('shape = {}'.format(str(tensor.shape)))

        # print('---------Relu---------')
        # for x in tf.get_default_graph().get_operations() :
        #     if  "Relu" in x.type:
        #         print('name = {}'.format( x.name ) )
        #         tensor = tf.get_default_graph().get_tensor_by_name(x.name+":0")
        #         print('shape = {}'.format(str(tensor.shape)))

        # print('---------tf.LOCAL_VARIABLES()---------')
        # get_collection_var = tf.get_collection(tf.GraphKeys.LOCAL_VARIABLES)
        # for v in get_collection_var:
        #     print(v)

        # print('---------tf.LOCAL_RESOURCES()---------')
        # get_collection_var = tf.get_collection(tf.GraphKeys.LOCAL_RESOURCES)
        # for v in get_collection_var:
        #     print(v)

        # print('---------tf.GLOBAL_STEP()---------')
        # get_collection_var = tf.get_collection(tf.GraphKeys.GLOBAL_STEP)
        # for v in get_collection_var:
        #     print(v)
        print('END=============print_all_var(), %s===============' % head_str)