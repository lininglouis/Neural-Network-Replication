# siamese figure shows two network with same weights. As for the implementation, we only use one network, and pass the two images two times through the this network.

# siamese如何处理两个网络（其实是一个网络）
# siamese如何处理输入的batch的，应该是n个pair加起来的batch，然后从batch里面依次拿pair喂给siamese network
# contrastive loss如何计算的
