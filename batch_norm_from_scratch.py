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


def net2(X, debug=False):
    ########################
    #  Define the computation of the first convolutional layer
    ########################
    h1_conv = nd.Convolution(data=X, weight=W1, bias=b1, kernel=(3,3), num_filter=20)
    h1_normed = batch_norm(h1_conv, gamma1, beta1, scope_name='bn1', is_training=True)
    h1_activation = relu(h1_normed)
    h1 = nd.Pooling(data=h1_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h1 shape: %s" % (np.array(h1.shape)))

    ########################
    #  Define the computation of the second convolutional layer
    ########################
    h2_conv = nd.Convolution(data=h1, weight=W2, bias=b2, kernel=(5,5), num_filter=50)
    h2_normed = batch_norm(h2_conv, gamma2, beta2, scope_name='bn2', is_training=is_training)
    h2_activation = relu(h2_normed)
    h2 = nd.Pooling(data=h2_activation, pool_type="avg", kernel=(2,2), stride=(2,2))
    if debug:
        print("h2 shape: %s" % (np.array(h2.shape)))

    ########################
    #  Flattening h2 so that we can feed it into a fully-connected layer
    ########################
    h2 = nd.flatten(h2)
    if debug:
        print("Flat h2 shape: %s" % (np.array(h2.shape)))

    ########################
    #  Define the computation of the third (fully-connected) layer
    ########################
    h3_linear = nd.dot(h2, W3) + b3
    h3_normed = batch_norm(h3_linear, gamma3, beta3, scope_name='bn3', is_training=is_training)
    h3 = relu(h3_normed)
    if debug:
        print("h3 shape: %s" % (np.array(h3.shape)))

    ########################
    #  Define the computation of the output layer
    ########################
    yhat_linear = nd.dot(h3, W4) + b4
    if debug:
        print("yhat_linear shape: %s" % (np.array(yhat_linear.shape)))

    return yhat_linear





mx.random.seed(1)
ctx = mx.cpu()

batch_size = 64
num_inputs = 784
num_outputs=10


train_data = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=True, transform=transform),
                                      batch_size, shuffle=True)

test_data  = mx.gluon.data.DataLoader(mx.gluon.data.vision.MNIST(train=False, transform=transform),
                                      batch_size, shuffle=True)


# net = gluon.nn.Sequential()
# net.add(gluon.nn.Conv2D(channels=20, kernel_size=3))
# net.add(gluon.nn.BatchNorm())
# net.add(gluon.nn.Activation(activation='relu'))
# net.add(gluon.nn.Flatten())
# net.add(gluon.nn.Dense(units=10))
#net.collect_params().initialize(mx.init.Xavier(magnitude=2.24), ctx=ctx)



#optimizer
#trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': .1})
#softmax_cross_entropy = gluon.loss.SoftmaxCrossEntropyLoss()


# def softmax_cross_entropy(yhat_linear, y):
#     return - nd.nansum(y * nd.log_softmax(yhat_linear), axis=0, exclude=True)


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
 