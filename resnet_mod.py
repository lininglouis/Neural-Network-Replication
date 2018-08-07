
from __future__ import division

__all__ = ['ResNetV1', 'ResNetV2',
           'BasicBlockV1', 'BasicBlockV2',
           'BottleneckV1', 'BottleneckV2',
           'resnet18_v1', 'resnet34_v1', 'resnet50_v1', 'resnet101_v1', 'resnet152_v1',
           'resnet18_v2', 'resnet34_v2', 'resnet50_v2', 'resnet101_v2', 'resnet152_v2',
           'get_resnet']

import os
import mxnet as mx
from mxnet import nd, autograd, gluon, init
from mxnet.gluon import nn




class Resnet101(gluon.HybridBlock):
    def __init__(self, **kwargs):
        super(Resnet101, self).__init__(**kwargs)

        with self.name_scope():
            self.conv1 = self.get_conv1()
            self.conv2 = self.conv_block(conv_num=2, block_repeat_num=3)
            self.conv3 = self.conv_block(conv_num=3, block_repeat_num=4)
            self.conv4 = self.conv_block(conv_num=4, block_repeat_num=23)
            self.conv5 = self.conv_block(conv_num=5, block_repeat_num=3)

    def get_conv1(self):
        self.block = nn.HybridSequential(prefix='conv1_')
        self.block.add( nn.BatchNorm(scale=False, epsilon=2e-5, use_global_stats=True))
        self.block.add( nn.Conv2D(channels=64, kernel_size=7,padding=2, strides=3, activation='relu'))
        self.block.add( nn.BatchNorm() )
        self.block.add( nn.Activation('relu') )
        self.block.add( nn.MaxPool2D(pool_size=3, strides=2, padding=1)) 
        return self.block
 
 
    def _conv_bn(self, kernel_size, channels, prefix):
        block = nn.HybridSequential(prefix=prefix)
        block.add( nn.Conv2D(channels=channels, kernel_size=kernel_size, padding=1, strides=1, activation='relu'))
        block.add( nn.BatchNorm() )
        block.add( nn.Activation('relu') )
        return block

    def conv_block(self, **kwargs):
        conv_num = kwargs['conv_num']
        block_repeat_num = kwargs['block_repeat_num']
        low_channels = 64  * (2 ** (conv_num-2))
        high_channels = 256 * (2 ** (conv_num-2))

        conv_blk = nn.HybridSequential(prefix='CONV_{}'.format(conv_num))
        conv_blk.add(Bottleneck( block_num=conv_num,
                                 branch_name = 'a',
                                 low_channels = low_channels, 
                                 high_channels=high_channels, 
                                 upsample=True))   
        
        for sub_block_num in range(block_repeat_num-1):
            conv_blk.add(Bottleneck(block_num=conv_num, 
                                    branch_name = 'b',
                                    low_channels=low_channels, 
                                    high_channels=high_channels,
                                    upsample=False))
        return conv_blk


    def hybrid_forward(self, F, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x




class Bottleneck(gluon.HybridBlock):
    def __init__(self, upsample, block_num, branch_name, low_channels, high_channels, **kwargs):
        super(Bottleneck, self).__init__(**kwargs)
        prefix =  "res{0}{1}".format(block_num, branch_name) 

        self.upsample = upsample
        if self.upsample:
            pass

            self.conv_bn_otherbranch = self._conv_bn(kernel_size=1, channels=int(256*2**(block_num-2)), prefix= prefix +'_branch1')

        self.conv_bn_relu1 = self._conv_bn_relu(kernel_size=1, channels=low_channels,  prefix = prefix + '_branch2a')
        self.conv_bn_relu2 = self._conv_bn_relu(kernel_size=3, channels=low_channels,  prefix = prefix + '_branch2b', padding=1)
        self.conv_bn3      = self._conv_bn     (kernel_size=1, channels=high_channels, prefix = prefix +'_branch2c')



    def _conv_bn_relu(self, kernel_size, channels, prefix, padding=0):
        block = self._conv_bn(kernel_size, channels, prefix, padding)
        block.add( nn.Activation('relu') )
        return block

    def _conv_bn(self, kernel_size, channels, prefix, padding=0):
        block = nn.HybridSequential(prefix=prefix)
        block.add( nn.Conv2D(channels=channels, kernel_size=kernel_size, padding=padding, strides=1, activation='relu'))
        block.add( nn.BatchNorm() )
        block.add( nn.Activation('relu') )
        return block

    def hybrid_forward(self, F, x):
        residual = x
        if self.upsample:
            residual = self.conv_bn_otherbranch(x)
        
        x = self.conv_bn_relu1(x)       #1x1 64
        x = self.conv_bn_relu2(x)       # 3x3 64
        x = self.conv_bn3(x)            #1x1 256 
        return residual+x



if __name__ == '__main__': 

    data = nd.random.uniform(shape=(64, 3, 224, 224))
    net = Resnet101()
    net.initialize()
    res = net(data)

    print(net)

 
