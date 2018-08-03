import mxnet as mx
from mxnet import nd, autograd, gluon
from mxnet.gluon import nn
import numpy as np 



def transform(data, label):
    return nd.transpose(data=data.astype(np.float32), axes=(2,0,1))/255, label.astype(np.float32)
    
def evaluate_accuracy(net, data_iter):

    acc = mx.metric.Accuracy()

    for i, (data, label) in enumerate(data_iter):
        data = data.as_in_context(ctx)
        target_label= label.as_in_context(ctx)

        pred = net(data)
        pred_label = nd.argmax(pred, axis=1)

        acc.update(preds=pred_label, labels=target_label)

    return acc.get()
 
def batch_norm(data, gamma, beta, scope_name, is_training, eps=1e-5, momentum=0.9,):

    # gamma and beta 's shape  is  same as the channel counts
    global _BN_MOVING_MEANS
    global _BN_MOVING_STDS

    N, C, H, W = data.shape
    _mean = nd.mean(data, axis=(0, 2, 3)).reshape((1, C, 1, 1))
    _variance =  nd.mean( (data - _mean)**2, axis=(0,2,3)).reshape((1, C, 1, 1))
    _std = nd.sqrt(_variance)

    if is_training: 
        X_normed = (data - _mean ) / (_std + eps)
    else:
        X_normed = (data - _BN_MOVING_MEANS[scope_name]['mean']) / (_BN_MOVING_STDS[scope_name]['std'] + eps)

    X_output = gamma.reshape((1,C,1,1)) * X_normed + beta.reshape((1,C,1,1))


    if scope_name not in _BN_MOVING_MEANS:
        _BN_MOVING_MEANS[scope_name] = _mean
    else:
        _BN_MOVING_MEANS[scope_name] = momentum * _BN_MOVING_MEANS[scope_name] + (1-momentum) * _mean 

    if scope_name not in _BN_MOVING_STDS:
        _BN_MOVING_STDS[scope_name] = _std     
    else:
        _BN_MOVING_STDS[scope_name] = momentum * _BN_MOVING_STDS[scope_name]   + (1-momentum) * _std 

    return X_output
  
def net(X):

    std = 0.01
    h1_conv   = mx.nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3), num_filter=20)
    h1_normed = batch_norm(data= h1_conv, gamma=gamma1, beta=beta1, is_training=True, scope_name='bn1')
    h1_relu = mx.nd.relu(h1_normed)
    h1_pool = mx.nd.Pooling(data=h1_relu, pool_type='avg', kernel=(2,2), stride=(2,2))

    h2 = mx.nd.flatten(h1_pool)
    h3 = mx.nd.dot(h2, W2) + b2

    return h3


 


mx.random.seed(1)
ctx = mx.cpu()

batch_size = 64
num_inputs = 784
num_outputs=10


train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)

test_data  = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                      batch_size, shuffle=True)

  
epochs=2
_BN_MOVING_MEANS = {}
_BN_MOVING_STDS = {}

std = 0.01
W1 = nd.random_normal(shape=(20,1,3,3), scale=std, ctx=ctx)
b1 = nd.random_normal(shape=20, scale=std, ctx=ctx)
gamma1 = nd.random_normal(shape=20, loc=1, scale=std, ctx=ctx) 
beta1  = nd.random_normal(shape=20, scale=std, ctx=ctx)
W2 = nd.random_normal(shape=(3380, 10), scale=std, ctx=ctx)
b2 = nd.random_normal(shape=10, scale=std, ctx=ctx)

params = [W1, b1, gamma1, beta1, W2, b2]
for param in params:
    param.attach_grad()

lr = .001


 
epochs = 1
moving_loss = 0.
learning_rate = .001

def SGD(params, lr):
    for param in params:
        param[:] = param - lr * param.grad
 

def softmax_cross_entropy(yhat_linear, y):
    return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)


for e in range(epochs):
    for i, (data, label) in enumerate(train_data):
        data = data.as_in_context(ctx)
        label = label.as_in_context(ctx)
        num_outputs=10
        label_one_hot = nd.one_hot(label, num_outputs)
        with autograd.record():
            # we are in training process,
            # so we normalize the data using batch mean and variance
            output = net(data)
            loss = softmax_cross_entropy(output, label_one_hot)
        loss.backward()
        SGD(params, learning_rate)

        if i == 0:
            moving_loss = nd.mean(loss).asscalar()
        else:
            moving_loss = .99 * moving_loss + .01 * nd.mean(loss).asscalar()
        print(moving_loss)


'''
for i in range(epochs):
    for idx, (data, label) in enumerate(train_data):

        if idx <20:
            data = data.as_in_context(mx.cpu())
            label = label.as_in_context(mx.cpu())

            with autograd.record():
                pred = net(data)
                loss = softmax_cross_entropy(pred, label)
            loss.backward()

            for param in params:
            	param[:] -= lr * param.grad

            batch_size = data.shape[0] 
            #trainer.step(batch_size)

            cur_loss = nd.mean(loss).asscalar()
            print('{:.2f}'.format(cur_loss))

    test_eval  = evaluate_accuracy(net, test_data)
    train_eval = evaluate_accuracy(net, train_data)

    print(test_eval)
    print('----------------------')
    print(train_eval)
#    print("epochs {:.2f} train acc: {:.2f}  test acc: {:.2f} ".format(i, train_eval, test_eval))


'''
 
