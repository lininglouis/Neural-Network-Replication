import _pickle as cPickle
import mxnet as mx
from mxnet import nd, gluon
 




def get_resnet_v1_conv4(data):
    eps = 1e-5
    conv1 = mx.nd.Convolution(name='conv1', data=data, num_filter=64, pad=(3, 3), kernel=(7, 7), stride=(2, 2), no_bias=True)
    bn_conv1 = mx.nd.BatchNorm(name='bn_conv1', data=conv1, use_global_stats=True, fix_gamma=False, eps=eps)
    scale_conv1 = bn_conv1
    conv1_relu = mx.nd.Activation(name='conv1_relu', data=scale_conv1, act_type='relu')
    pool1 = mx.nd.Pooling(name='pool1', data=conv1_relu, pooling_convention='full', pad=(0, 0), kernel=(3, 3),
                              stride=(2, 2), pool_type='max')

    
    res2a_branch1 = mx.nd.Convolution(name='res2a_branch1', data=pool1, num_filter=256, pad=(0, 0), kernel=(1, 1),
                                          stride=(1, 1), no_bias=True)
    bn2a_branch1 = mx.nd.BatchNorm(name='bn2a_branch1', data=res2a_branch1, use_global_stats=True, fix_gamma=False, eps=eps)
    scale2a_branch1 = bn2a_branch1


    res2a_branch2a = mx.nd.Convolution(name='res2a_branch2a', data=pool1, num_filter=64, pad=(0, 0), kernel=(1, 1),
                                           stride=(1, 1), no_bias=True)
    bn2a_branch2a = mx.nd.BatchNorm(name='bn2a_branch2a', data=res2a_branch2a, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale2a_branch2a = bn2a_branch2a
    res2a_branch2a_relu = mx.nd.Activation(name='res2a_branch2a_relu', data=scale2a_branch2a, act_type='relu')


    res2a_branch2b = mx.nd.Convolution(name='res2a_branch2b', data=res2a_branch2a_relu, num_filter=64, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn2a_branch2b = mx.nd.BatchNorm(name='bn2a_branch2b', data=res2a_branch2b, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale2a_branch2b = bn2a_branch2b
    res2a_branch2b_relu = mx.nd.Activation(name='res2a_branch2b_relu', data=scale2a_branch2b, act_type='relu')
    res2a_branch2c = mx.nd.Convolution(name='res2a_branch2c', data=res2a_branch2b_relu, num_filter=256, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn2a_branch2c = mx.nd.BatchNorm(name='bn2a_branch2c', data=res2a_branch2c, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale2a_branch2c = bn2a_branch2c

 

    res2a = mx.nd.broadcast_add(name='res2a', *[scale2a_branch1, scale2a_branch2c])
    res2a_relu = mx.nd.Activation(name='res2a_relu', data=res2a, act_type='relu')
    res2b_branch2a = mx.nd.Convolution(name='res2b_branch2a', data=res2a_relu, num_filter=64, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn2b_branch2a = mx.nd.BatchNorm(name='bn2b_branch2a', data=res2b_branch2a, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale2b_branch2a = bn2b_branch2a
    res2b_branch2a_relu = mx.nd.Activation(name='res2b_branch2a_relu', data=scale2b_branch2a, act_type='relu')
    res2b_branch2b = mx.nd.Convolution(name='res2b_branch2b', data=res2b_branch2a_relu, num_filter=64, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn2b_branch2b = mx.nd.BatchNorm(name='bn2b_branch2b', data=res2b_branch2b, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale2b_branch2b = bn2b_branch2b
    res2b_branch2b_relu = mx.nd.Activation(name='res2b_branch2b_relu', data=scale2b_branch2b, act_type='relu')
    res2b_branch2c = mx.nd.Convolution(name='res2b_branch2c', data=res2b_branch2b_relu, num_filter=256, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn2b_branch2c = mx.nd.BatchNorm(name='bn2b_branch2c', data=res2b_branch2c, use_global_stats=True,
                                        fix_gamma=False, eps=eps)


    
    scale2b_branch2c = bn2b_branch2c
    res2b = mx.nd.broadcast_add(name='res2b', *[res2a_relu, scale2b_branch2c])
    res2b_relu = mx.nd.Activation(name='res2b_relu', data=res2b, act_type='relu')
    res2c_branch2a = mx.nd.Convolution(name='res2c_branch2a', data=res2b_relu, num_filter=64, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn2c_branch2a = mx.nd.BatchNorm(name='bn2c_branch2a', data=res2c_branch2a, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale2c_branch2a = bn2c_branch2a
    res2c_branch2a_relu = mx.nd.Activation(name='res2c_branch2a_relu', data=scale2c_branch2a, act_type='relu')
    res2c_branch2b = mx.nd.Convolution(name='res2c_branch2b', data=res2c_branch2a_relu, num_filter=64, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn2c_branch2b = mx.nd.BatchNorm(name='bn2c_branch2b', data=res2c_branch2b, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale2c_branch2b = bn2c_branch2b
    res2c_branch2b_relu = mx.nd.Activation(name='res2c_branch2b_relu', data=scale2c_branch2b, act_type='relu')
    res2c_branch2c = mx.nd.Convolution(name='res2c_branch2c', data=res2c_branch2b_relu, num_filter=256, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn2c_branch2c = mx.nd.BatchNorm(name='bn2c_branch2c', data=res2c_branch2c, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale2c_branch2c = bn2c_branch2c
    res2c = mx.nd.broadcast_add(name='res2c', *[res2b_relu, scale2c_branch2c])
    res2c_relu = mx.nd.Activation(name='res2c_relu', data=res2c, act_type='relu')
    res3a_branch1 = mx.nd.Convolution(name='res3a_branch1', data=res2c_relu, num_filter=512, pad=(0, 0),
                                          kernel=(1, 1), stride=(2, 2), no_bias=True)
    bn3a_branch1 = mx.nd.BatchNorm(name='bn3a_branch1', data=res3a_branch1, use_global_stats=True, fix_gamma=False, eps=eps)
    scale3a_branch1 = bn3a_branch1
    res3a_branch2a = mx.nd.Convolution(name='res3a_branch2a', data=res2c_relu, num_filter=128, pad=(0, 0),
                                           kernel=(1, 1), stride=(2, 2), no_bias=True)
    bn3a_branch2a = mx.nd.BatchNorm(name='bn3a_branch2a', data=res3a_branch2a, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale3a_branch2a = bn3a_branch2a
    res3a_branch2a_relu = mx.nd.Activation(name='res3a_branch2a_relu', data=scale3a_branch2a, act_type='relu')
    res3a_branch2b = mx.nd.Convolution(name='res3a_branch2b', data=res3a_branch2a_relu, num_filter=128, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn3a_branch2b = mx.nd.BatchNorm(name='bn3a_branch2b', data=res3a_branch2b, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale3a_branch2b = bn3a_branch2b
    res3a_branch2b_relu = mx.nd.Activation(name='res3a_branch2b_relu', data=scale3a_branch2b, act_type='relu')
    res3a_branch2c = mx.nd.Convolution(name='res3a_branch2c', data=res3a_branch2b_relu, num_filter=512, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn3a_branch2c = mx.nd.BatchNorm(name='bn3a_branch2c', data=res3a_branch2c, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale3a_branch2c = bn3a_branch2c
    res3a = mx.nd.broadcast_add(name='res3a', *[scale3a_branch1, scale3a_branch2c])
    res3a_relu = mx.nd.Activation(name='res3a_relu', data=res3a, act_type='relu')
    res3b1_branch2a = mx.nd.Convolution(name='res3b1_branch2a', data=res3a_relu, num_filter=128, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn3b1_branch2a = mx.nd.BatchNorm(name='bn3b1_branch2a', data=res3b1_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b1_branch2a = bn3b1_branch2a
    res3b1_branch2a_relu = mx.nd.Activation(name='res3b1_branch2a_relu', data=scale3b1_branch2a, act_type='relu')
    res3b1_branch2b = mx.nd.Convolution(name='res3b1_branch2b', data=res3b1_branch2a_relu, num_filter=128,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn3b1_branch2b = mx.nd.BatchNorm(name='bn3b1_branch2b', data=res3b1_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b1_branch2b = bn3b1_branch2b
    res3b1_branch2b_relu = mx.nd.Activation(name='res3b1_branch2b_relu', data=scale3b1_branch2b, act_type='relu')
    res3b1_branch2c = mx.nd.Convolution(name='res3b1_branch2c', data=res3b1_branch2b_relu, num_filter=512,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn3b1_branch2c = mx.nd.BatchNorm(name='bn3b1_branch2c', data=res3b1_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b1_branch2c = bn3b1_branch2c
    res3b1 = mx.nd.broadcast_add(name='res3b1', *[res3a_relu, scale3b1_branch2c])
    res3b1_relu = mx.nd.Activation(name='res3b1_relu', data=res3b1, act_type='relu')
    res3b2_branch2a = mx.nd.Convolution(name='res3b2_branch2a', data=res3b1_relu, num_filter=128, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn3b2_branch2a = mx.nd.BatchNorm(name='bn3b2_branch2a', data=res3b2_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b2_branch2a = bn3b2_branch2a
    res3b2_branch2a_relu = mx.nd.Activation(name='res3b2_branch2a_relu', data=scale3b2_branch2a, act_type='relu')
    res3b2_branch2b = mx.nd.Convolution(name='res3b2_branch2b', data=res3b2_branch2a_relu, num_filter=128,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn3b2_branch2b = mx.nd.BatchNorm(name='bn3b2_branch2b', data=res3b2_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b2_branch2b = bn3b2_branch2b
    res3b2_branch2b_relu = mx.nd.Activation(name='res3b2_branch2b_relu', data=scale3b2_branch2b, act_type='relu')
    res3b2_branch2c = mx.nd.Convolution(name='res3b2_branch2c', data=res3b2_branch2b_relu, num_filter=512,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn3b2_branch2c = mx.nd.BatchNorm(name='bn3b2_branch2c', data=res3b2_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b2_branch2c = bn3b2_branch2c
    res3b2 = mx.nd.broadcast_add(name='res3b2', *[res3b1_relu, scale3b2_branch2c])
    res3b2_relu = mx.nd.Activation(name='res3b2_relu', data=res3b2, act_type='relu')
    res3b3_branch2a = mx.nd.Convolution(name='res3b3_branch2a', data=res3b2_relu, num_filter=128, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn3b3_branch2a = mx.nd.BatchNorm(name='bn3b3_branch2a', data=res3b3_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b3_branch2a = bn3b3_branch2a
    res3b3_branch2a_relu = mx.nd.Activation(name='res3b3_branch2a_relu', data=scale3b3_branch2a, act_type='relu')
    res3b3_branch2b = mx.nd.Convolution(name='res3b3_branch2b', data=res3b3_branch2a_relu, num_filter=128,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn3b3_branch2b = mx.nd.BatchNorm(name='bn3b3_branch2b', data=res3b3_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b3_branch2b = bn3b3_branch2b
    res3b3_branch2b_relu = mx.nd.Activation(name='res3b3_branch2b_relu', data=scale3b3_branch2b, act_type='relu')
    res3b3_branch2c = mx.nd.Convolution(name='res3b3_branch2c', data=res3b3_branch2b_relu, num_filter=512,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn3b3_branch2c = mx.nd.BatchNorm(name='bn3b3_branch2c', data=res3b3_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale3b3_branch2c = bn3b3_branch2c
    res3b3 = mx.nd.broadcast_add(name='res3b3', *[res3b2_relu, scale3b3_branch2c])
    res3b3_relu = mx.nd.Activation(name='res3b3_relu', data=res3b3, act_type='relu')
    res4a_branch1 = mx.nd.Convolution(name='res4a_branch1', data=res3b3_relu, num_filter=1024, pad=(0, 0),
                                          kernel=(1, 1), stride=(2, 2), no_bias=True)
    bn4a_branch1 = mx.nd.BatchNorm(name='bn4a_branch1', data=res4a_branch1, use_global_stats=True, fix_gamma=False, eps=eps)
    scale4a_branch1 = bn4a_branch1
    res4a_branch2a = mx.nd.Convolution(name='res4a_branch2a', data=res3b3_relu, num_filter=256, pad=(0, 0),
                                           kernel=(1, 1), stride=(2, 2), no_bias=True)
    bn4a_branch2a = mx.nd.BatchNorm(name='bn4a_branch2a', data=res4a_branch2a, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale4a_branch2a = bn4a_branch2a
    res4a_branch2a_relu = mx.nd.Activation(name='res4a_branch2a_relu', data=scale4a_branch2a, act_type='relu')
    res4a_branch2b = mx.nd.Convolution(name='res4a_branch2b', data=res4a_branch2a_relu, num_filter=256, pad=(1, 1),
                                           kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4a_branch2b = mx.nd.BatchNorm(name='bn4a_branch2b', data=res4a_branch2b, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale4a_branch2b = bn4a_branch2b
    res4a_branch2b_relu = mx.nd.Activation(name='res4a_branch2b_relu', data=scale4a_branch2b, act_type='relu')
    res4a_branch2c = mx.nd.Convolution(name='res4a_branch2c', data=res4a_branch2b_relu, num_filter=1024, pad=(0, 0),
                                           kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4a_branch2c = mx.nd.BatchNorm(name='bn4a_branch2c', data=res4a_branch2c, use_global_stats=True,
                                        fix_gamma=False, eps=eps)
    scale4a_branch2c = bn4a_branch2c
    res4a = mx.nd.broadcast_add(name='res4a', *[scale4a_branch1, scale4a_branch2c])
    res4a_relu = mx.nd.Activation(name='res4a_relu', data=res4a, act_type='relu')
    res4b1_branch2a = mx.nd.Convolution(name='res4b1_branch2a', data=res4a_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b1_branch2a = mx.nd.BatchNorm(name='bn4b1_branch2a', data=res4b1_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b1_branch2a = bn4b1_branch2a
    res4b1_branch2a_relu = mx.nd.Activation(name='res4b1_branch2a_relu', data=scale4b1_branch2a, act_type='relu')
    res4b1_branch2b = mx.nd.Convolution(name='res4b1_branch2b', data=res4b1_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b1_branch2b = mx.nd.BatchNorm(name='bn4b1_branch2b', data=res4b1_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b1_branch2b = bn4b1_branch2b
    res4b1_branch2b_relu = mx.nd.Activation(name='res4b1_branch2b_relu', data=scale4b1_branch2b, act_type='relu')
    res4b1_branch2c = mx.nd.Convolution(name='res4b1_branch2c', data=res4b1_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b1_branch2c = mx.nd.BatchNorm(name='bn4b1_branch2c', data=res4b1_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b1_branch2c = bn4b1_branch2c
    res4b1 = mx.nd.broadcast_add(name='res4b1', *[res4a_relu, scale4b1_branch2c])
    res4b1_relu = mx.nd.Activation(name='res4b1_relu', data=res4b1, act_type='relu')
    res4b2_branch2a = mx.nd.Convolution(name='res4b2_branch2a', data=res4b1_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b2_branch2a = mx.nd.BatchNorm(name='bn4b2_branch2a', data=res4b2_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b2_branch2a = bn4b2_branch2a
    res4b2_branch2a_relu = mx.nd.Activation(name='res4b2_branch2a_relu', data=scale4b2_branch2a, act_type='relu')
    res4b2_branch2b = mx.nd.Convolution(name='res4b2_branch2b', data=res4b2_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b2_branch2b = mx.nd.BatchNorm(name='bn4b2_branch2b', data=res4b2_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b2_branch2b = bn4b2_branch2b
    res4b2_branch2b_relu = mx.nd.Activation(name='res4b2_branch2b_relu', data=scale4b2_branch2b, act_type='relu')
    res4b2_branch2c = mx.nd.Convolution(name='res4b2_branch2c', data=res4b2_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b2_branch2c = mx.nd.BatchNorm(name='bn4b2_branch2c', data=res4b2_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b2_branch2c = bn4b2_branch2c
    res4b2 = mx.nd.broadcast_add(name='res4b2', *[res4b1_relu, scale4b2_branch2c])
    res4b2_relu = mx.nd.Activation(name='res4b2_relu', data=res4b2, act_type='relu')
    res4b3_branch2a = mx.nd.Convolution(name='res4b3_branch2a', data=res4b2_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b3_branch2a = mx.nd.BatchNorm(name='bn4b3_branch2a', data=res4b3_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b3_branch2a = bn4b3_branch2a
    res4b3_branch2a_relu = mx.nd.Activation(name='res4b3_branch2a_relu', data=scale4b3_branch2a, act_type='relu')
    res4b3_branch2b = mx.nd.Convolution(name='res4b3_branch2b', data=res4b3_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b3_branch2b = mx.nd.BatchNorm(name='bn4b3_branch2b', data=res4b3_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b3_branch2b = bn4b3_branch2b
    res4b3_branch2b_relu = mx.nd.Activation(name='res4b3_branch2b_relu', data=scale4b3_branch2b, act_type='relu')
    res4b3_branch2c = mx.nd.Convolution(name='res4b3_branch2c', data=res4b3_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b3_branch2c = mx.nd.BatchNorm(name='bn4b3_branch2c', data=res4b3_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b3_branch2c = bn4b3_branch2c
    res4b3 = mx.nd.broadcast_add(name='res4b3', *[res4b2_relu, scale4b3_branch2c])
    res4b3_relu = mx.nd.Activation(name='res4b3_relu', data=res4b3, act_type='relu')
    res4b4_branch2a = mx.nd.Convolution(name='res4b4_branch2a', data=res4b3_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b4_branch2a = mx.nd.BatchNorm(name='bn4b4_branch2a', data=res4b4_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b4_branch2a = bn4b4_branch2a
    res4b4_branch2a_relu = mx.nd.Activation(name='res4b4_branch2a_relu', data=scale4b4_branch2a, act_type='relu')
    res4b4_branch2b = mx.nd.Convolution(name='res4b4_branch2b', data=res4b4_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b4_branch2b = mx.nd.BatchNorm(name='bn4b4_branch2b', data=res4b4_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b4_branch2b = bn4b4_branch2b
    res4b4_branch2b_relu = mx.nd.Activation(name='res4b4_branch2b_relu', data=scale4b4_branch2b, act_type='relu')
    res4b4_branch2c = mx.nd.Convolution(name='res4b4_branch2c', data=res4b4_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b4_branch2c = mx.nd.BatchNorm(name='bn4b4_branch2c', data=res4b4_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b4_branch2c = bn4b4_branch2c
    res4b4 = mx.nd.broadcast_add(name='res4b4', *[res4b3_relu, scale4b4_branch2c])
    res4b4_relu = mx.nd.Activation(name='res4b4_relu', data=res4b4, act_type='relu')
    res4b5_branch2a = mx.nd.Convolution(name='res4b5_branch2a', data=res4b4_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b5_branch2a = mx.nd.BatchNorm(name='bn4b5_branch2a', data=res4b5_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b5_branch2a = bn4b5_branch2a
    res4b5_branch2a_relu = mx.nd.Activation(name='res4b5_branch2a_relu', data=scale4b5_branch2a, act_type='relu')
    res4b5_branch2b = mx.nd.Convolution(name='res4b5_branch2b', data=res4b5_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b5_branch2b = mx.nd.BatchNorm(name='bn4b5_branch2b', data=res4b5_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b5_branch2b = bn4b5_branch2b
    res4b5_branch2b_relu = mx.nd.Activation(name='res4b5_branch2b_relu', data=scale4b5_branch2b, act_type='relu')
    res4b5_branch2c = mx.nd.Convolution(name='res4b5_branch2c', data=res4b5_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b5_branch2c = mx.nd.BatchNorm(name='bn4b5_branch2c', data=res4b5_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b5_branch2c = bn4b5_branch2c
    res4b5 = mx.nd.broadcast_add(name='res4b5', *[res4b4_relu, scale4b5_branch2c])
    res4b5_relu = mx.nd.Activation(name='res4b5_relu', data=res4b5, act_type='relu')
    res4b6_branch2a = mx.nd.Convolution(name='res4b6_branch2a', data=res4b5_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b6_branch2a = mx.nd.BatchNorm(name='bn4b6_branch2a', data=res4b6_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b6_branch2a = bn4b6_branch2a
    res4b6_branch2a_relu = mx.nd.Activation(name='res4b6_branch2a_relu', data=scale4b6_branch2a, act_type='relu')
    res4b6_branch2b = mx.nd.Convolution(name='res4b6_branch2b', data=res4b6_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b6_branch2b = mx.nd.BatchNorm(name='bn4b6_branch2b', data=res4b6_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b6_branch2b = bn4b6_branch2b
    res4b6_branch2b_relu = mx.nd.Activation(name='res4b6_branch2b_relu', data=scale4b6_branch2b, act_type='relu')
    res4b6_branch2c = mx.nd.Convolution(name='res4b6_branch2c', data=res4b6_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b6_branch2c = mx.nd.BatchNorm(name='bn4b6_branch2c', data=res4b6_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b6_branch2c = bn4b6_branch2c
    res4b6 = mx.nd.broadcast_add(name='res4b6', *[res4b5_relu, scale4b6_branch2c])
    res4b6_relu = mx.nd.Activation(name='res4b6_relu', data=res4b6, act_type='relu')
    res4b7_branch2a = mx.nd.Convolution(name='res4b7_branch2a', data=res4b6_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b7_branch2a = mx.nd.BatchNorm(name='bn4b7_branch2a', data=res4b7_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b7_branch2a = bn4b7_branch2a
    res4b7_branch2a_relu = mx.nd.Activation(name='res4b7_branch2a_relu', data=scale4b7_branch2a, act_type='relu')
    res4b7_branch2b = mx.nd.Convolution(name='res4b7_branch2b', data=res4b7_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b7_branch2b = mx.nd.BatchNorm(name='bn4b7_branch2b', data=res4b7_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b7_branch2b = bn4b7_branch2b
    res4b7_branch2b_relu = mx.nd.Activation(name='res4b7_branch2b_relu', data=scale4b7_branch2b, act_type='relu')
    res4b7_branch2c = mx.nd.Convolution(name='res4b7_branch2c', data=res4b7_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b7_branch2c = mx.nd.BatchNorm(name='bn4b7_branch2c', data=res4b7_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b7_branch2c = bn4b7_branch2c
    res4b7 = mx.nd.broadcast_add(name='res4b7', *[res4b6_relu, scale4b7_branch2c])
    res4b7_relu = mx.nd.Activation(name='res4b7_relu', data=res4b7, act_type='relu')
    res4b8_branch2a = mx.nd.Convolution(name='res4b8_branch2a', data=res4b7_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b8_branch2a = mx.nd.BatchNorm(name='bn4b8_branch2a', data=res4b8_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b8_branch2a = bn4b8_branch2a
    res4b8_branch2a_relu = mx.nd.Activation(name='res4b8_branch2a_relu', data=scale4b8_branch2a, act_type='relu')
    res4b8_branch2b = mx.nd.Convolution(name='res4b8_branch2b', data=res4b8_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b8_branch2b = mx.nd.BatchNorm(name='bn4b8_branch2b', data=res4b8_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b8_branch2b = bn4b8_branch2b
    res4b8_branch2b_relu = mx.nd.Activation(name='res4b8_branch2b_relu', data=scale4b8_branch2b, act_type='relu')
    res4b8_branch2c = mx.nd.Convolution(name='res4b8_branch2c', data=res4b8_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b8_branch2c = mx.nd.BatchNorm(name='bn4b8_branch2c', data=res4b8_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b8_branch2c = bn4b8_branch2c
    res4b8 = mx.nd.broadcast_add(name='res4b8', *[res4b7_relu, scale4b8_branch2c])
    res4b8_relu = mx.nd.Activation(name='res4b8_relu', data=res4b8, act_type='relu')
    res4b9_branch2a = mx.nd.Convolution(name='res4b9_branch2a', data=res4b8_relu, num_filter=256, pad=(0, 0),
                                            kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b9_branch2a = mx.nd.BatchNorm(name='bn4b9_branch2a', data=res4b9_branch2a, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b9_branch2a = bn4b9_branch2a
    res4b9_branch2a_relu = mx.nd.Activation(name='res4b9_branch2a_relu', data=scale4b9_branch2a, act_type='relu')
    res4b9_branch2b = mx.nd.Convolution(name='res4b9_branch2b', data=res4b9_branch2a_relu, num_filter=256,
                                            pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b9_branch2b = mx.nd.BatchNorm(name='bn4b9_branch2b', data=res4b9_branch2b, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b9_branch2b = bn4b9_branch2b
    res4b9_branch2b_relu = mx.nd.Activation(name='res4b9_branch2b_relu', data=scale4b9_branch2b, act_type='relu')
    res4b9_branch2c = mx.nd.Convolution(name='res4b9_branch2c', data=res4b9_branch2b_relu, num_filter=1024,
                                            pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b9_branch2c = mx.nd.BatchNorm(name='bn4b9_branch2c', data=res4b9_branch2c, use_global_stats=True,
                                         fix_gamma=False, eps=eps)
    scale4b9_branch2c = bn4b9_branch2c
    res4b9 = mx.nd.broadcast_add(name='res4b9', *[res4b8_relu, scale4b9_branch2c])
    res4b9_relu = mx.nd.Activation(name='res4b9_relu', data=res4b9, act_type='relu')
    res4b10_branch2a = mx.nd.Convolution(name='res4b10_branch2a', data=res4b9_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b10_branch2a = mx.nd.BatchNorm(name='bn4b10_branch2a', data=res4b10_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b10_branch2a = bn4b10_branch2a
    res4b10_branch2a_relu = mx.nd.Activation(name='res4b10_branch2a_relu', data=scale4b10_branch2a, act_type='relu')
    res4b10_branch2b = mx.nd.Convolution(name='res4b10_branch2b', data=res4b10_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b10_branch2b = mx.nd.BatchNorm(name='bn4b10_branch2b', data=res4b10_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b10_branch2b = bn4b10_branch2b
    res4b10_branch2b_relu = mx.nd.Activation(name='res4b10_branch2b_relu', data=scale4b10_branch2b, act_type='relu')
    res4b10_branch2c = mx.nd.Convolution(name='res4b10_branch2c', data=res4b10_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b10_branch2c = mx.nd.BatchNorm(name='bn4b10_branch2c', data=res4b10_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b10_branch2c = bn4b10_branch2c
    res4b10 = mx.nd.broadcast_add(name='res4b10', *[res4b9_relu, scale4b10_branch2c])
    res4b10_relu = mx.nd.Activation(name='res4b10_relu', data=res4b10, act_type='relu')
    res4b11_branch2a = mx.nd.Convolution(name='res4b11_branch2a', data=res4b10_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b11_branch2a = mx.nd.BatchNorm(name='bn4b11_branch2a', data=res4b11_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b11_branch2a = bn4b11_branch2a
    res4b11_branch2a_relu = mx.nd.Activation(name='res4b11_branch2a_relu', data=scale4b11_branch2a, act_type='relu')
    res4b11_branch2b = mx.nd.Convolution(name='res4b11_branch2b', data=res4b11_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b11_branch2b = mx.nd.BatchNorm(name='bn4b11_branch2b', data=res4b11_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b11_branch2b = bn4b11_branch2b
    res4b11_branch2b_relu = mx.nd.Activation(name='res4b11_branch2b_relu', data=scale4b11_branch2b, act_type='relu')
    res4b11_branch2c = mx.nd.Convolution(name='res4b11_branch2c', data=res4b11_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b11_branch2c = mx.nd.BatchNorm(name='bn4b11_branch2c', data=res4b11_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b11_branch2c = bn4b11_branch2c
    res4b11 = mx.nd.broadcast_add(name='res4b11', *[res4b10_relu, scale4b11_branch2c])
    res4b11_relu = mx.nd.Activation(name='res4b11_relu', data=res4b11, act_type='relu')
    res4b12_branch2a = mx.nd.Convolution(name='res4b12_branch2a', data=res4b11_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b12_branch2a = mx.nd.BatchNorm(name='bn4b12_branch2a', data=res4b12_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b12_branch2a = bn4b12_branch2a
    res4b12_branch2a_relu = mx.nd.Activation(name='res4b12_branch2a_relu', data=scale4b12_branch2a, act_type='relu')
    res4b12_branch2b = mx.nd.Convolution(name='res4b12_branch2b', data=res4b12_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b12_branch2b = mx.nd.BatchNorm(name='bn4b12_branch2b', data=res4b12_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b12_branch2b = bn4b12_branch2b
    res4b12_branch2b_relu = mx.nd.Activation(name='res4b12_branch2b_relu', data=scale4b12_branch2b, act_type='relu')
    res4b12_branch2c = mx.nd.Convolution(name='res4b12_branch2c', data=res4b12_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b12_branch2c = mx.nd.BatchNorm(name='bn4b12_branch2c', data=res4b12_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b12_branch2c = bn4b12_branch2c
    res4b12 = mx.nd.broadcast_add(name='res4b12', *[res4b11_relu, scale4b12_branch2c])
    res4b12_relu = mx.nd.Activation(name='res4b12_relu', data=res4b12, act_type='relu')
    res4b13_branch2a = mx.nd.Convolution(name='res4b13_branch2a', data=res4b12_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b13_branch2a = mx.nd.BatchNorm(name='bn4b13_branch2a', data=res4b13_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b13_branch2a = bn4b13_branch2a
    res4b13_branch2a_relu = mx.nd.Activation(name='res4b13_branch2a_relu', data=scale4b13_branch2a, act_type='relu')
    res4b13_branch2b = mx.nd.Convolution(name='res4b13_branch2b', data=res4b13_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b13_branch2b = mx.nd.BatchNorm(name='bn4b13_branch2b', data=res4b13_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b13_branch2b = bn4b13_branch2b
    res4b13_branch2b_relu = mx.nd.Activation(name='res4b13_branch2b_relu', data=scale4b13_branch2b, act_type='relu')
    res4b13_branch2c = mx.nd.Convolution(name='res4b13_branch2c', data=res4b13_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b13_branch2c = mx.nd.BatchNorm(name='bn4b13_branch2c', data=res4b13_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b13_branch2c = bn4b13_branch2c
    res4b13 = mx.nd.broadcast_add(name='res4b13', *[res4b12_relu, scale4b13_branch2c])
    res4b13_relu = mx.nd.Activation(name='res4b13_relu', data=res4b13, act_type='relu')
    res4b14_branch2a = mx.nd.Convolution(name='res4b14_branch2a', data=res4b13_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b14_branch2a = mx.nd.BatchNorm(name='bn4b14_branch2a', data=res4b14_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b14_branch2a = bn4b14_branch2a
    res4b14_branch2a_relu = mx.nd.Activation(name='res4b14_branch2a_relu', data=scale4b14_branch2a, act_type='relu')
    res4b14_branch2b = mx.nd.Convolution(name='res4b14_branch2b', data=res4b14_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b14_branch2b = mx.nd.BatchNorm(name='bn4b14_branch2b', data=res4b14_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b14_branch2b = bn4b14_branch2b
    res4b14_branch2b_relu = mx.nd.Activation(name='res4b14_branch2b_relu', data=scale4b14_branch2b, act_type='relu')
    res4b14_branch2c = mx.nd.Convolution(name='res4b14_branch2c', data=res4b14_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b14_branch2c = mx.nd.BatchNorm(name='bn4b14_branch2c', data=res4b14_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b14_branch2c = bn4b14_branch2c
    res4b14 = mx.nd.broadcast_add(name='res4b14', *[res4b13_relu, scale4b14_branch2c])
    res4b14_relu = mx.nd.Activation(name='res4b14_relu', data=res4b14, act_type='relu')
    res4b15_branch2a = mx.nd.Convolution(name='res4b15_branch2a', data=res4b14_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b15_branch2a = mx.nd.BatchNorm(name='bn4b15_branch2a', data=res4b15_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b15_branch2a = bn4b15_branch2a
    res4b15_branch2a_relu = mx.nd.Activation(name='res4b15_branch2a_relu', data=scale4b15_branch2a, act_type='relu')
    res4b15_branch2b = mx.nd.Convolution(name='res4b15_branch2b', data=res4b15_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b15_branch2b = mx.nd.BatchNorm(name='bn4b15_branch2b', data=res4b15_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b15_branch2b = bn4b15_branch2b
    res4b15_branch2b_relu = mx.nd.Activation(name='res4b15_branch2b_relu', data=scale4b15_branch2b, act_type='relu')
    res4b15_branch2c = mx.nd.Convolution(name='res4b15_branch2c', data=res4b15_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b15_branch2c = mx.nd.BatchNorm(name='bn4b15_branch2c', data=res4b15_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b15_branch2c = bn4b15_branch2c
    res4b15 = mx.nd.broadcast_add(name='res4b15', *[res4b14_relu, scale4b15_branch2c])
    res4b15_relu = mx.nd.Activation(name='res4b15_relu', data=res4b15, act_type='relu')
    res4b16_branch2a = mx.nd.Convolution(name='res4b16_branch2a', data=res4b15_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b16_branch2a = mx.nd.BatchNorm(name='bn4b16_branch2a', data=res4b16_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b16_branch2a = bn4b16_branch2a
    res4b16_branch2a_relu = mx.nd.Activation(name='res4b16_branch2a_relu', data=scale4b16_branch2a, act_type='relu')
    res4b16_branch2b = mx.nd.Convolution(name='res4b16_branch2b', data=res4b16_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b16_branch2b = mx.nd.BatchNorm(name='bn4b16_branch2b', data=res4b16_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b16_branch2b = bn4b16_branch2b
    res4b16_branch2b_relu = mx.nd.Activation(name='res4b16_branch2b_relu', data=scale4b16_branch2b, act_type='relu')
    res4b16_branch2c = mx.nd.Convolution(name='res4b16_branch2c', data=res4b16_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b16_branch2c = mx.nd.BatchNorm(name='bn4b16_branch2c', data=res4b16_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b16_branch2c = bn4b16_branch2c
    res4b16 = mx.nd.broadcast_add(name='res4b16', *[res4b15_relu, scale4b16_branch2c])
    res4b16_relu = mx.nd.Activation(name='res4b16_relu', data=res4b16, act_type='relu')
    res4b17_branch2a = mx.nd.Convolution(name='res4b17_branch2a', data=res4b16_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b17_branch2a = mx.nd.BatchNorm(name='bn4b17_branch2a', data=res4b17_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b17_branch2a = bn4b17_branch2a
    res4b17_branch2a_relu = mx.nd.Activation(name='res4b17_branch2a_relu', data=scale4b17_branch2a, act_type='relu')
    res4b17_branch2b = mx.nd.Convolution(name='res4b17_branch2b', data=res4b17_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b17_branch2b = mx.nd.BatchNorm(name='bn4b17_branch2b', data=res4b17_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b17_branch2b = bn4b17_branch2b
    res4b17_branch2b_relu = mx.nd.Activation(name='res4b17_branch2b_relu', data=scale4b17_branch2b, act_type='relu')
    res4b17_branch2c = mx.nd.Convolution(name='res4b17_branch2c', data=res4b17_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b17_branch2c = mx.nd.BatchNorm(name='bn4b17_branch2c', data=res4b17_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b17_branch2c = bn4b17_branch2c
    res4b17 = mx.nd.broadcast_add(name='res4b17', *[res4b16_relu, scale4b17_branch2c])
    res4b17_relu = mx.nd.Activation(name='res4b17_relu', data=res4b17, act_type='relu')
    res4b18_branch2a = mx.nd.Convolution(name='res4b18_branch2a', data=res4b17_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b18_branch2a = mx.nd.BatchNorm(name='bn4b18_branch2a', data=res4b18_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b18_branch2a = bn4b18_branch2a
    res4b18_branch2a_relu = mx.nd.Activation(name='res4b18_branch2a_relu', data=scale4b18_branch2a, act_type='relu')
    res4b18_branch2b = mx.nd.Convolution(name='res4b18_branch2b', data=res4b18_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b18_branch2b = mx.nd.BatchNorm(name='bn4b18_branch2b', data=res4b18_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b18_branch2b = bn4b18_branch2b
    res4b18_branch2b_relu = mx.nd.Activation(name='res4b18_branch2b_relu', data=scale4b18_branch2b, act_type='relu')
    res4b18_branch2c = mx.nd.Convolution(name='res4b18_branch2c', data=res4b18_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b18_branch2c = mx.nd.BatchNorm(name='bn4b18_branch2c', data=res4b18_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b18_branch2c = bn4b18_branch2c
    res4b18 = mx.nd.broadcast_add(name='res4b18', *[res4b17_relu, scale4b18_branch2c])
    res4b18_relu = mx.nd.Activation(name='res4b18_relu', data=res4b18, act_type='relu')
    res4b19_branch2a = mx.nd.Convolution(name='res4b19_branch2a', data=res4b18_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b19_branch2a = mx.nd.BatchNorm(name='bn4b19_branch2a', data=res4b19_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b19_branch2a = bn4b19_branch2a
    res4b19_branch2a_relu = mx.nd.Activation(name='res4b19_branch2a_relu', data=scale4b19_branch2a, act_type='relu')
    res4b19_branch2b = mx.nd.Convolution(name='res4b19_branch2b', data=res4b19_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b19_branch2b = mx.nd.BatchNorm(name='bn4b19_branch2b', data=res4b19_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b19_branch2b = bn4b19_branch2b
    res4b19_branch2b_relu = mx.nd.Activation(name='res4b19_branch2b_relu', data=scale4b19_branch2b, act_type='relu')
    res4b19_branch2c = mx.nd.Convolution(name='res4b19_branch2c', data=res4b19_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b19_branch2c = mx.nd.BatchNorm(name='bn4b19_branch2c', data=res4b19_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b19_branch2c = bn4b19_branch2c
    res4b19 = mx.nd.broadcast_add(name='res4b19', *[res4b18_relu, scale4b19_branch2c])
    res4b19_relu = mx.nd.Activation(name='res4b19_relu', data=res4b19, act_type='relu')
    res4b20_branch2a = mx.nd.Convolution(name='res4b20_branch2a', data=res4b19_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b20_branch2a = mx.nd.BatchNorm(name='bn4b20_branch2a', data=res4b20_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b20_branch2a = bn4b20_branch2a
    res4b20_branch2a_relu = mx.nd.Activation(name='res4b20_branch2a_relu', data=scale4b20_branch2a, act_type='relu')
    res4b20_branch2b = mx.nd.Convolution(name='res4b20_branch2b', data=res4b20_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b20_branch2b = mx.nd.BatchNorm(name='bn4b20_branch2b', data=res4b20_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b20_branch2b = bn4b20_branch2b
    res4b20_branch2b_relu = mx.nd.Activation(name='res4b20_branch2b_relu', data=scale4b20_branch2b, act_type='relu')
    res4b20_branch2c = mx.nd.Convolution(name='res4b20_branch2c', data=res4b20_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b20_branch2c = mx.nd.BatchNorm(name='bn4b20_branch2c', data=res4b20_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b20_branch2c = bn4b20_branch2c
    res4b20 = mx.nd.broadcast_add(name='res4b20', *[res4b19_relu, scale4b20_branch2c])
    res4b20_relu = mx.nd.Activation(name='res4b20_relu', data=res4b20, act_type='relu')
    res4b21_branch2a = mx.nd.Convolution(name='res4b21_branch2a', data=res4b20_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b21_branch2a = mx.nd.BatchNorm(name='bn4b21_branch2a', data=res4b21_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b21_branch2a = bn4b21_branch2a
    res4b21_branch2a_relu = mx.nd.Activation(name='res4b21_branch2a_relu', data=scale4b21_branch2a, act_type='relu')
    res4b21_branch2b = mx.nd.Convolution(name='res4b21_branch2b', data=res4b21_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b21_branch2b = mx.nd.BatchNorm(name='bn4b21_branch2b', data=res4b21_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b21_branch2b = bn4b21_branch2b
    res4b21_branch2b_relu = mx.nd.Activation(name='res4b21_branch2b_relu', data=scale4b21_branch2b, act_type='relu')
    res4b21_branch2c = mx.nd.Convolution(name='res4b21_branch2c', data=res4b21_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b21_branch2c = mx.nd.BatchNorm(name='bn4b21_branch2c', data=res4b21_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b21_branch2c = bn4b21_branch2c
    res4b21 = mx.nd.broadcast_add(name='res4b21', *[res4b20_relu, scale4b21_branch2c])
    res4b21_relu = mx.nd.Activation(name='res4b21_relu', data=res4b21, act_type='relu')
    res4b22_branch2a = mx.nd.Convolution(name='res4b22_branch2a', data=res4b21_relu, num_filter=256, pad=(0, 0),
                                             kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b22_branch2a = mx.nd.BatchNorm(name='bn4b22_branch2a', data=res4b22_branch2a, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b22_branch2a = bn4b22_branch2a
    res4b22_branch2a_relu = mx.nd.Activation(name='res4b22_branch2a_relu', data=scale4b22_branch2a, act_type='relu')
    res4b22_branch2b = mx.nd.Convolution(name='res4b22_branch2b', data=res4b22_branch2a_relu, num_filter=256,
                                             pad=(1, 1), kernel=(3, 3), stride=(1, 1), no_bias=True)
    bn4b22_branch2b = mx.nd.BatchNorm(name='bn4b22_branch2b', data=res4b22_branch2b, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b22_branch2b = bn4b22_branch2b
    res4b22_branch2b_relu = mx.nd.Activation(name='res4b22_branch2b_relu', data=scale4b22_branch2b, act_type='relu')
    res4b22_branch2c = mx.nd.Convolution(name='res4b22_branch2c', data=res4b22_branch2b_relu, num_filter=1024,
                                             pad=(0, 0), kernel=(1, 1), stride=(1, 1), no_bias=True)
    bn4b22_branch2c = mx.nd.BatchNorm(name='bn4b22_branch2c', data=res4b22_branch2c, use_global_stats=True,
                                          fix_gamma=False, eps=eps)
    scale4b22_branch2c = bn4b22_branch2c
    res4b22 = mx.nd.broadcast_add(name='res4b22', *[res4b21_relu, scale4b22_branch2c])
    res4b22_relu = mx.nd.Activation(name='res4b22_relu', data=res4b22, act_type='relu')
    return res4b22_relu, res4b22, res3b3



 


data = mx.nd.random.uniform(0,1, (2, 3,224,224) )
conv_feat, res4b22, res3b3 = get_resnet_v1_conv4(data) 

arg_shape, out_shape, aux_shape  = conv_feat.infer_shape(data=(8,3,224,224))

