import torch
from dataset import load_data_imdb
from bi_rnn import BiRNN
from text_cnn import TextCNN


def predict_sentiment(net, vocab, sequence):
    """预测文本序列的情感"""
    sequence = torch.tensor(vocab[sequence.split()])
    label = torch.argmax(net(sequence.reshape(1, -1)), dim=1)
    return 'positive' if label == 1 else 'negative'


def predict_test(model_name):
    batch_size = 64
    train_iter, test_iter, vocab = load_data_imdb(batch_size)
    embed_size = 100
    if model_name == 'BiRNN':
        num_hiddens, num_layers = 100, 2
        net = BiRNN(len(vocab), embed_size, num_hiddens, num_layers)
    elif model_name == 'TextCNN':
        kernel_sizes, nums_channels = [3, 4, 5], [100, 100, 100]
        net = TextCNN(len(vocab), embed_size, kernel_sizes, nums_channels)
    else:
        raise ValueError('模型命名错误')
    net.load_state_dict(torch.load(f'../model/{type(net).__name__}_model.pth'))
    print(predict_sentiment(net, vocab, 'this movie is so great'))
    print(predict_sentiment(net, vocab, 'this movie is so bad'))


if __name__ == '__main__':
    # for name in ('BiRNN', 'TextCNN'):
    #     predict_test(name)
        predict_test('BiRNN')