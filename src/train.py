import torch
from torch import nn
from d2l import torch as d2l
from dataset import load_data_imdb
from bi_rnn import BiRNN, biRNNModel
from text_cnn import TextCNN, textCNNModel
from deal_fig import save_fig


def training(net, train_iter, test_iter, num_epochs, optimizer, loss, device):
    ''' 训练迭代，将输出动画并保存走势图 '''
    timer, num_batches = d2l.Timer(), len(train_iter)
    optimizer = optimizer
    loss = loss
    animator = d2l.Animator(xlabel='轮次', xlim=[1, num_epochs],ylim=[0,1],
                            legend=['训练损失', '训练准确率', '测试准确率'])
    net.to(device)
    for epoch in range(num_epochs):
        # Sum of training loss, sum of training accuracy, no. of examples, no. of predictions
        metric = d2l.Accumulator(4)
        net.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            X, y = X.to(device), y.to(device)
            y_hat = net(X)
            l = loss(y_hat, y)
            l.sum().backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(l.sum(), d2l.accuracy(y_hat, y), X.shape[0], y.numel())
            timer.stop()
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (metric[0] / metric[2], metric[1] / metric[3],
                              None))
        test_acc = d2l.evaluate_accuracy_gpu(net, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'训练损失 {metric[0] / metric[2]:.3f}, 训练准确率 '
          f'{metric[1] / metric[3]:.3f}, 测试准确率 {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} 样本每秒 '
          f'{str(device)}')
    save_fig(animator.axes, type(net).__name__)


def trained_net(model_name):
    ''' 超参数设置和训练方法选择，保存并返回训练好的模型'''
    # 加载数据集
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb(batch_size)
    # 模型获取
    if model_name == 'BiRNN':
        net = biRNNModel(len(vocab))
    elif model_name == 'TextCNN':
        net = textCNNModel(len(vocab))
    else:
        raise ValueError('模型命名错误')
    # 加载预训练的词向量
    glove_embedding = d2l.TokenEmbedding('glove.6b.100d')
    embeds = glove_embedding[vocab.idx_to_token]
    net.embedding.weight.data.copy_(embeds)
    net.embedding.weight.requires_grad = False
    # 训练参数初始化
    if isinstance(net, BiRNN):
        lr, num_epochs = 0.01, 10
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss(reduction="none")
        device = d2l.try_gpu()
    elif isinstance(net, TextCNN):
        lr, num_epochs = 0.01, 10  # textCNN的lr设为0.001
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
        loss = nn.CrossEntropyLoss(reduction="none")
        device = d2l.try_gpu()
    # 训练模型
    training(net, train_iter, test_iter, num_epochs, optimizer, loss, device)
    # 保存模型
    torch.save(net.state_dict(), f'../model/{type(net).__name__}_model.pth')
    return net


if __name__ == '__main__':
    # for model_name in ('BiRNN', 'TextCNN'):
    #     trained_net(model_name)
    # trained_net('BiRNN')
    trained_net('TextCNN')
