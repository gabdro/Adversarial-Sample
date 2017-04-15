import numpy as np
import chainer
from chainer import cuda, Function, gradient_check, report, training, utils, Variable
from chainer import datasets, iterators, optimizers, serializers
from chainer import Link, Chain, ChainList
import chainer.functions as F
import chainer.links as L
from chainer.training import extensions

#ネットワーク構築
class CIFAR10_cifer(Chain):
    def __init__(self):
        super(CIFAR10_cifer , self).__init__(
                                             #ネットワークの記述
                                             conv1=F.Convolution2D(3,64,3,pad=1),
                                             conv2=F.Convolution2D(64,128,3,pad=1),
                                             conv3=F.Convolution2D(128,128,3,pad=1),
                                             conv4=F.Convolution2D(128,256,3,pad=1),
                                             conv5=F.Convolution2D(256,256,3,pad=1),
                                             conv6=F.Convolution2D(256,256,3,pad=1),
                                             conv7=F.Convolution2D(256,256,3,pad=1),
                                             conv8=F.Convolution2D(256,256,3,pad=1),
                                             l2 = F.Linear(4096,2304),
                                             l3 = F.Linear(2304,2304),
                                             l4=F.Linear(2304,10)
                                             )

    def __call__(self , x,train=True):
        h = F.relu(self.conv1(x))
        h = F.max_pooling_2d(h, ksize = 5, stride = 2, pad =2)
        h = F.relu(self.conv2(h))
        h = F.max_pooling_2d(h, ksize = 5, stride = 2, pad =2)
        h = F.relu(self.conv3(h))
        h = F.max_pooling_2d(h, ksize = 5, stride = 2, pad =2)
        h = F.relu(self.conv4(h))
        h = F.max_pooling_2d(h, ksize = 5, stride = 2, pad =2)
        h = F.relu(self.l3(h))
        #h = F.dropout(h, train=train)
        y = self.l4(h)
        return y

class Classifier(Chain):
    def __init__(self,predictor):
        super(Classifier,self).__init__(predictor=predictor)

    def __call__(self,x,t):
        y = self.predictor(x)
        loss = F.softmax_cross_entropy(y,t)
        accuracy = F.accuracy(y,t)
        report({'loss': loss, 'accuracy': accuracy},self)
        return loss
